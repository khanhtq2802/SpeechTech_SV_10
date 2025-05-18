import os
import time
import argparse
# import csv # Không dùng csv.reader nữa
from glob import glob
from collections import defaultdict

import torch
import torchaudio
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import roc_curve

# --- Các hàm và lớp được giữ lại hoặc điều chỉnh từ train.py ---

# Hàm tính EER (cập nhật để xử lý các trường hợp biên)
def compute_eer(labels, scores):
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    if len(labels_arr) == 0:
        print("Cảnh báo: Không có nhãn nào để tính EER. Trả về EER=1.0 (tệ nhất).")
        return 1.0
    
    unique_labels = np.unique(labels_arr)
    if len(unique_labels) < 2:
        # Nếu chỉ có một lớp, không thể tính đường cong ROC một cách có ý nghĩa.
        # EER=0.5 thường được coi là hiệu suất của bộ phân loại ngẫu nhiên.
        print(f"Cảnh báo: Chỉ có một lớp trong nhãn ({unique_labels}). Không thể tính ROC. Trả về EER=0.5.")
        return 0.5 

    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr, pos_label=1)
    fnr = 1 - tpr

    if len(fpr) == 0 or len(tpr) == 0:
        print("Cảnh báo: fpr hoặc tpr rỗng sau roc_curve. Trả về EER=0.5.")
        return 0.5

    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_index]
    return eer


# Mô hình chính (điều chỉnh để không có loss_fn)
class SVModel_Inference(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.proj = nn.Linear(self.wav2vec.config.hidden_size, 256)

    def forward(self, input_values, attention_mask):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask).last_hidden_state
        emb = outputs.mean(dim=1)
        proj = self.proj(emb)
        return proj

# Hàm trích xuất embedding cho một file âm thanh
def get_embedding_for_file(audio_path, model, feature_extractor, device):
    model.eval()
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = wav.squeeze(0) 

    inputs = feature_extractor(wav, sampling_rate=16000, return_tensors='pt')
    input_values = inputs.input_values 

    if input_values.ndim == 1:
        input_values = input_values.unsqueeze(0) 
    
    input_values = input_values.to(device)
    
    # Tạo attention mask. Giả định giá trị padding là 0,
    # nhất quán với collate_fn trong train.py.
    attention_mask = (input_values != 0).long().to(device) # <<< SỬA LỖI Ở ĐÂY

    with torch.no_grad():
        embedding = model(input_values, attention_mask) 
    return embedding.squeeze(0).cpu().numpy() 


# --- Hàm chính cho public_test.py ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='/ASV/dataset/public_test',
                        help="Thư mục chứa tập public test (bao gồm thư mục wav và test_list_gt.csv)")
    parser.add_argument('--model_path', type=str, default='/ASV/old/asv_epoch3.pt',
                        help="Đường dẫn đến file trọng số mô hình đã huấn luyện (.pt)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo Feature Extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base', sampling_rate=16_000)

    # 2. Khởi tạo mô hình
    model = SVModel_Inference().to(device)

    # 3. Load trọng số mô hình
    print(f"Đang tải trọng số từ: {args.model_path}")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False) 
        print("Tải trọng số thành công.")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file trọng số tại: {args.model_path}")
        exit(1)
    except Exception as e:
        print(f"LỖI: Không thể tải trọng số từ {args.model_path}. Lỗi: {e}")
        exit(1)
        
    model.eval()

    # 4. Đọc file test_list_gt.csv
    test_list_path = os.path.join(args.test_dir, 'test_list_gt.csv')
    pairs = []
    unique_wav_files = set()
    print(f"Đang đọc danh sách kiểm tra từ: {test_list_path}")
    
    if not os.path.exists(test_list_path):
        print(f"LỖI: Không tìm thấy file danh sách kiểm tra: {test_list_path}")
        exit(1)

    with open(test_list_path, 'r', encoding='utf-8') as f:
        for line_num, line_content in enumerate(f):
            line_stripped = line_content.strip() 
            if not line_stripped: 
                continue

            if line_stripped.startswith('"') and line_stripped.endswith('"'):
                line_unquoted = line_stripped[1:-1]
            else:
                line_unquoted = line_stripped 

            parts = line_unquoted.split('\t') 

            if len(parts) == 3:
                try:
                    label_str = parts[0]
                    path1_relative = parts[1]
                    path2_relative = parts[2]

                    label = int(label_str)
                    path1 = os.path.join(args.test_dir, path1_relative)
                    path2 = os.path.join(args.test_dir, path2_relative)

                    pairs.append({'label': label, 'path1': path1, 'path2': path2})
                    unique_wav_files.add(path1)
                    unique_wav_files.add(path2)
                except ValueError:
                    print(f"CẢNH BÁO: Không thể chuyển đổi label '{label_str}' sang int ở dòng {line_num+1}: '{line_content.strip()}'")
                except Exception as e:
                    print(f"Lỗi khi xử lý dòng {line_num+1}: '{line_content.strip()}' - {e}")
            else:
                if line_num < 5 : 
                     print(f"CẢNH BÁO: Dòng không được phân tích thành 3 phần (dòng {line_num+1}): '{line_stripped}'. Số phần: {len(parts)}. Nội dung các phần: {parts}")
    
    if not pairs:
        print("LỖI: Không tìm thấy cặp kiểm tra nào hợp lệ từ file. Vui lòng kiểm tra định dạng file test_list_gt.csv.")
        exit(1)
        
    print(f"Tìm thấy {len(pairs)} cặp kiểm tra và {len(unique_wav_files)} file âm thanh duy nhất.")

    # 5. Trích xuất embeddings cho tất cả các file âm thanh duy nhất
    print("Đang trích xuất embeddings cho các file âm thanh...")
    audio_embeddings = {}
    for audio_file in tqdm(list(unique_wav_files), desc="Trích xuất embeddings"):
        if not os.path.exists(audio_file):
            print(f"CẢNH BÁO: File không tồn tại {audio_file}, sẽ bỏ qua các cặp liên quan.")
            audio_embeddings[audio_file] = None 
            continue
        try:
            emb = get_embedding_for_file(audio_file, model, feature_extractor, device)
            audio_embeddings[audio_file] = emb
        except Exception as e:
            print(f"Lỗi khi xử lý file {audio_file}: {e}")
            audio_embeddings[audio_file] = None

    # 6. Tính toán cosine similarity cho từng cặp và thu thập labels
    print("Đang tính toán cosine similarity và EER...")
    scores = []
    ground_truths = []

    for pair_info in tqdm(pairs, desc="Tính similarity"):
        emb1 = audio_embeddings.get(pair_info['path1'])
        emb2 = audio_embeddings.get(pair_info['path2'])

        if emb1 is None or emb2 is None:
            continue 

        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        scores.append(sim)
        ground_truths.append(pair_info['label'])

    if not ground_truths or not scores:
        print("Không có cặp nào được xử lý thành công, không thể tính EER.")
    else:
        # 7. Tính EER
        eer = compute_eer(ground_truths, scores)
        print(f"Equal Error Rate (EER) trên tập public test: {eer:.4f}")