import os
import time
import argparse
from glob import glob
from collections import defaultdict

import torch
import torchaudio
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm.auto import tqdm
import numpy as np
# sklearn.metrics.roc_curve không cần thiết vì chúng ta không tính EER ở đây

# --- Các hàm và lớp được giữ lại hoặc điều chỉnh từ public_test.py ---

# Mô hình chính (giống như trong public_test.py)
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

# Hàm trích xuất embedding cho một file âm thanh (giống như trong public_test.py đã sửa lỗi)
def get_embedding_for_file(audio_path, model, feature_extractor, device):
    model.eval()
    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"LỖI: Không thể tải file audio: {audio_path}. Lỗi: {e}")
        return None # Trả về None nếu không tải được file

    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = wav.squeeze(0) 

    try:
        inputs = feature_extractor(wav, sampling_rate=16000, return_tensors='pt')
    except Exception as e:
        # Một số file wav rất ngắn có thể gây lỗi ở đây nếu wav rỗng sau resample/squeeze
        print(f"LỖI: Không thể trích xuất đặc trưng cho file: {audio_path}. Lỗi: {e}")
        return None
        
    input_values = inputs.input_values 

    if input_values.ndim == 1:
        input_values = input_values.unsqueeze(0) 
    
    input_values = input_values.to(device)
    
    # Tạo attention mask. Giả định giá trị padding là 0.
    attention_mask = (input_values != 0).long().to(device)

    with torch.no_grad():
        embedding = model(input_values, attention_mask) 
    return embedding.squeeze(0).cpu().numpy() 


# --- Hàm chính cho private_test.py ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate similarity scores for private test set.")
    parser.add_argument('--private_test_dir', type=str, default='/ASV/dataset/private_test',
                        help="Thư mục gốc chứa dữ liệu private test (ví dụ: /ASV/dataset/private_test).")
    parser.add_argument('--audio_folder_name', type=str, default='private-test-data-sv',
                        help="Tên thư mục con chứa các file .wav (ví dụ: private-test-data-sv).")
    parser.add_argument('--prompts_file_name', type=str, default='prompts_sv.csv',
                        help="Tên file chứa danh sách các cặp audio (ví dụ: prompts_sv.csv).")
    parser.add_argument('--model_path', type=str, default='/ASV/old/asv_epoch3.pt', # Nhớ thay đổi nếu cần
                        help="Đường dẫn đến file trọng số mô hình đã huấn luyện (.pt).")
    parser.add_argument('--output_file', type=str, default='predictions.txt',
                        help="Tên file output chứa các điểm similarity (mặc định: predictions.txt).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo Feature Extractor
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base', sampling_rate=16_000)
    except Exception as e:
        print(f"LỖI: Không thể khởi tạo Wav2Vec2FeatureExtractor. Lỗi: {e}")
        exit(1)

    # 2. Khởi tạo mô hình
    model = SVModel_Inference().to(device)

    # 3. Load trọng số mô hình
    print(f"Đang tải trọng số từ: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"LỖI: Không tìm thấy file trọng số tại: {args.model_path}")
        exit(1)
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False) 
        print("Tải trọng số thành công.")
    except Exception as e:
        print(f"LỖI: Không thể tải trọng số từ {args.model_path}. Lỗi: {e}")
        exit(1)
    model.eval()

    # 4. Đọc file prompts_sv.csv
    prompts_path = os.path.join(args.private_test_dir, args.prompts_file_name)
    audio_base_path = os.path.join(args.private_test_dir, args.audio_folder_name)
    
    trial_pairs_info = [] # Sẽ chứa các dict {'path1': ..., 'path2': ...}
    unique_wav_files = set()
    
    print(f"Đang đọc danh sách các cặp thử nghiệm từ: {prompts_path}")
    if not os.path.exists(prompts_path):
        print(f"LỖI: Không tìm thấy file prompts: {prompts_path}")
        exit(1)

    with open(prompts_path, 'r', encoding='utf-8') as f_prompts:
        for line_num, line_content in enumerate(f_prompts):
            line_stripped = line_content.strip()
            if not line_stripped:
                continue # Bỏ qua dòng trống

            parts = line_stripped.split() # Tách bằng khoảng trắng
            if len(parts) == 2:
                file1_basename = parts[0]
                file2_basename = parts[1]

                path1 = os.path.join(audio_base_path, file1_basename)
                path2 = os.path.join(audio_base_path, file2_basename)
                
                trial_pairs_info.append({'path1': path1, 'path2': path2, 'original_line': line_stripped})
                unique_wav_files.add(path1)
                unique_wav_files.add(path2)
            else:
                print(f"CẢNH BÁO: Dòng không hợp lệ (không có 2 phần) tại dòng {line_num+1} trong {prompts_path}: '{line_stripped}'")
    
    if not trial_pairs_info:
        print(f"LỖI: Không đọc được cặp thử nghiệm nào từ {prompts_path}. Vui lòng kiểm tra file.")
        exit(1)
    print(f"Tìm thấy {len(trial_pairs_info)} cặp thử nghiệm và {len(unique_wav_files)} file âm thanh duy nhất.")

    # 5. Trích xuất embeddings cho tất cả các file âm thanh duy nhất
    print("Đang trích xuất embeddings cho các file âm thanh...")
    audio_embeddings = {}
    for audio_file_path in tqdm(list(unique_wav_files), desc="Trích xuất embeddings"):
        if not os.path.exists(audio_file_path):
            print(f"CẢNH BÁO: File audio không tồn tại: {audio_file_path}. Embedding sẽ là None.")
            audio_embeddings[audio_file_path] = None 
            continue
        emb = get_embedding_for_file(audio_file_path, model, feature_extractor, device)
        audio_embeddings[audio_file_path] = emb # emb có thể là None nếu get_embedding_for_file thất bại

    # 6. Tính toán cosine similarity cho từng cặp và ghi vào file output
    print(f"Đang tính toán similarity scores và ghi vào file: {args.output_file}")
    
    num_errors_similarity = 0
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for pair_info in tqdm(trial_pairs_info, desc="Tính similarity và ghi file"):
            emb1 = audio_embeddings.get(pair_info['path1'])
            emb2 = audio_embeddings.get(pair_info['path2'])

            if emb1 is None or emb2 is None:
                # Nếu một trong hai embedding bị thiếu (do lỗi tải file hoặc trích xuất đặc trưng)
                # Ghi một giá trị mặc định (ví dụ: 0.0) vì file predictions.txt phải có đủ số dòng.
                # Cuộc thi không nói rõ xử lý thế nào, nên 0.0 là một lựa chọn an toàn.
                similarity_score = 0.0 
                if emb1 is None:
                    print(f"CẢNH BÁO: Thiếu embedding cho {pair_info['path1']} (cặp: {pair_info['original_line']}). Dùng score mặc định {similarity_score}.")
                if emb2 is None:
                    print(f"CẢNH BÁO: Thiếu embedding cho {pair_info['path2']} (cặp: {pair_info['original_line']}). Dùng score mặc định {similarity_score}.")
                num_errors_similarity +=1
            else:
                # Tính cosine similarity
                # np.linalg.norm có thể trả về 0 nếu embedding là vector 0 (hiếm nhưng có thể xảy ra)
                norm_emb1 = np.linalg.norm(emb1)
                norm_emb2 = np.linalg.norm(emb2)
                if norm_emb1 == 0 or norm_emb2 == 0:
                    similarity_score = 0.0 # Nếu một embedding là zero vector, sim là 0
                    print(f"CẢNH BÁO: Zero embedding cho {pair_info['path1']} hoặc {pair_info['path2']} (cặp: {pair_info['original_line']}). Dùng score mặc định {similarity_score}.")
                    num_errors_similarity +=1
                else:
                    similarity_score = np.dot(emb1, emb2) / (norm_emb1 * norm_emb2)
            
            f_out.write(f"{similarity_score:.8f}\n") # Ghi score với độ chính xác cao, theo sau là newline

    print(f"Hoàn thành! File '{args.output_file}' đã được tạo.")
    if num_errors_similarity > 0:
        print(f"LƯU Ý: Có {num_errors_similarity} cặp được gán score mặc định do lỗi embedding hoặc file audio.")