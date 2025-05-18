import os
import time
import argparse
from glob import glob
from collections import defaultdict

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import roc_curve

# Hàm tải metadata
def load_metadata(metadata_path):
    meta = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        headers = f.readline().strip().split()
        for line in f:
            parts = line.strip().split()
            if len(parts) != len(headers):
                continue
            info = dict(zip(headers, parts))
            meta[info['speaker_id']] = info
    return meta

# Hàm tải kết quả diarization
def load_diarization(diar_path):
    clean_set = set()
    with open(diar_path, 'r', encoding='utf-8') as f:
        for line in f:
            path, speakers = line.strip().split()
            if speakers == '1':
                clean_set.add(path)
    return clean_set

# Hàm tải kết quả phân loại giới tính và kiểm tra khớp với metadata
def load_gender(gender_path, speaker_meta):
    keep = set()
    gender_pred = {}
    with open(gender_path, 'r', encoding='utf-8') as f:
        for line in f:
            path, pred = line.strip().split()
            gender_pred[path] = pred

    for path, pred in gender_pred.items():
        speaker_id = path.split('/')[-2]  # Giả sử đường dẫn là /ASV/dataset/train_eval/data/id00000/00000.wav
        if speaker_id in speaker_meta:
            actual_gender = speaker_meta[speaker_id]['gender']
            if pred == actual_gender:
                keep.add(path)
    return keep

# Dataset cho dữ liệu âm thanh
class ASVDataset(Dataset):
    def __init__(self, data_dir, speaker_meta, clean_diar, clean_gender, ids):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base', sampling_rate=16_000)
        self.samples = []
        for spk in ids:
            spk_dir = os.path.join(data_dir, spk)
            if not os.path.isdir(spk_dir):
                continue
            wavs = glob(os.path.join(spk_dir, '*.wav'))
            wavs = [w for w in wavs if w in clean_diar and w in clean_gender]
            if len(wavs) < 2:
                continue
            for w in wavs:
                self.samples.append((w, spk))
        self.spk_to_idx = {s: i for i, s in enumerate(sorted({s for _, s in self.samples}))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, spk = self.samples[idx]
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        wav = wav.squeeze(0)
        inputs = self.feature_extractor(wav, sampling_rate=16000, return_tensors='pt')
        return inputs.input_values[0], self.spk_to_idx[spk]

# Hàm collate để xử lý batch
def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    attention = (inputs != 0).long()
    labels = torch.tensor(labels)
    return {'input_values': inputs, 'attention_mask': attention}, labels

# Lớp loss với margin
# To-do cần phải đổi lại tên CosFace
class AdditiveAngularMarginLoss(nn.Module):
    def __init__(self, in_features, out_features, margin=0.2, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        normed_emb = nn.functional.normalize(embeddings)
        normed_w = nn.functional.normalize(self.weight)
        cos = torch.matmul(normed_emb, normed_w.t())
        phi = cos - self.margin
        logits = torch.where(
            torch.arange(cos.size(1), device=cos.device).unsqueeze(0) == labels.unsqueeze(1),
            phi, cos
        )
        logits = logits * self.scale
        loss = self.ce(logits, labels)
        return loss

# Mô hình chính
class SVModel(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.proj = nn.Linear(self.wav2vec.config.hidden_size, 256)
        self.loss_fn = AdditiveAngularMarginLoss(256, num_speakers)

    def forward(self, input_values, attention_mask, labels=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask).last_hidden_state
        emb = outputs.mean(dim=1)
        proj = self.proj(emb)
        if labels is not None:
            loss = self.loss_fn(proj, labels)
            return loss, proj
        return proj

# Hàm tính EER
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer

# Hàm đánh giá EER trên tập dev
def evaluate_eer(model, dev_loader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch, batch_labels in tqdm(dev_loader, desc="Đánh giá embeddings"):
            input_values = batch['input_values'].cuda()
            attn = batch['attention_mask'].cuda()
            emb = model(input_values, attn)
            embeddings.append(emb.cpu().numpy())
            labels.append(batch_labels.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    same_pairs = []
    diff_pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                same_pairs.append((i, j))
            else:
                diff_pairs.append((i, j))

    # Điều chỉnh số lượng diff_pairs gấp 12 lần same_pairs
    num_same = len(same_pairs)
    num_diff = min(len(diff_pairs), 12 * num_same)
    if num_diff < len(diff_pairs):
        indices = np.random.choice(len(diff_pairs), num_diff, replace=False)
        diff_pairs = [diff_pairs[idx] for idx in indices]

    # Tính cosine similarity
    sims = []
    targets = []
    for i, j in same_pairs:
        sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
        sims.append(sim)
        targets.append(1)
    for i, j in diff_pairs:
        sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
        sims.append(sim)
        targets.append(0)

    eer = compute_eer(targets, sims)
    return eer

# Đếm tham số có thể huấn luyện
def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Hàm chính
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/ASV/dataset/train_eval/data')
    parser.add_argument('--meta', type=str, default='/ASV/dataset/train_eval/speaker-metadata.tsv')
    parser.add_argument('--diar', type=str, default='/ASV/dataset/train_eval/speaker-diarization-result.txt')
    parser.add_argument('--gender', type=str, default='/ASV/dataset/train_eval/voice-gender-classifier-result.txt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    args = parser.parse_args()

    # Tải dữ liệu
    speaker_meta = load_metadata(args.meta)
    clean_diar = load_diarization(args.diar)
    clean_gender = load_gender(args.gender, speaker_meta)  # Đã sửa để kiểm tra khớp giới tính

    # Chia tập train và dev
    all_ids = sorted(os.listdir(args.data_dir))
    train_ids = [i for i in all_ids if i <= 'id00899']
    dev_ids = [i for i in all_ids if i > 'id00899']

    train_ds = ASVDataset(args.data_dir, speaker_meta, clean_diar, clean_gender, train_ids)
    dev_ds = ASVDataset(args.data_dir, speaker_meta, clean_diar, clean_gender, dev_ids)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Khởi tạo mô hình
    model = SVModel(num_speakers=len(train_ds.spk_to_idx)).to("cuda")

    # Đóng băng các tham số không cần huấn luyện
    for name, p in model.named_parameters():
        if 'proj.weight' == name or 'loss_fn.weight' == name or name.endswith('bias'):
            p.requires_grad = True
        else:
            p.requires_grad = False

    print("Số tham số có thể huấn luyện:", count_trainable(model))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_eer = float('inf')
    no_improve = 0
    epoch = 0

    # Vòng lặp huấn luyện
    while True:
        epoch += 1

        # Giai đoạn train
        model.train()
        start = time.time()
        train_losses = []
        for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            input_values = batch['input_values'].cuda()
            attn = batch['attention_mask'].cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            loss, _ = model(input_values, attn, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_time = time.time() - start
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Thời gian: {train_time:.2f}s")

        # Giai đoạn validate
        eer = evaluate_eer(model, dev_loader)
        print(f"Epoch {epoch} - Dev EER: {eer:.4f}")

        # Lưu mô hình
        torch.save(model.state_dict(), f"asv_epoch{epoch}.pt")

        # Early stopping
        if eer < best_eer:
            best_eer = eer
            no_improve = 0
            torch.save(model.state_dict(), "asv_best.pt")
            print(f"Lưu mô hình tốt nhất với EER: {best_eer:.4f}")
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print("Dừng sớm do không cải thiện sau", args.patience, "epoch.")
            break