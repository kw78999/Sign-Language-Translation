# ⚙️ 0. 설치 및 임포트
import os
import json
import numpy as np
from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ✅ 단어 ID → 의미 매핑
WORD_LABELS = {
    "1506": "Miss",
    "1511": "Fishing",
    "1517": "Crybaby",
    "1518": "Sister"
}
HIDDEN_DIM = 20

from collections import Counter

def compute_class_weights(y_labels, num_classes):
    """레이블 벡터에서 클래스별 가중치를 계산하여 CrossEntropyLoss에 사용"""
    counts = Counter(y_labels)
    total = sum(counts.values())
    weights = [total / counts.get(i, 1) for i in range(num_classes)]  # count=0 회피 위해 +1 안함
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()  # 정규화 (선택 사항)
    return weights

def interpolate_sequence(seq, target_len=120):
    """
    seq: numpy array of shape (T, 274) where T is the number of frames
    target_len: desired sequence length (default 120)
    """
    current_len = len(seq)
    if current_len == target_len:
        return seq

    x_old = np.linspace(0, 1, current_len)
    x_new = np.linspace(0, 1, target_len)
    
    # 선형 보간
    interpolated_seq = np.zeros((target_len, seq.shape[1]))
    for i in range(seq.shape[1]):
        interpolated_seq[:, i] = np.interp(x_new, x_old, seq[:, i])

    return interpolated_seq


# 🚀 1. 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📂 2. JSON 파싱 함수 정의 (OpenPose 2D keypoint 전용)
def load_keypoint_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    keypoint_fields = ['pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    keypoints = []
    for field in keypoint_fields:
        raw = data['people'][field]
        arr = np.array(raw).reshape(-1, 3)[:, :2]  # (N, 2)
        keypoints.append(arr)

    # === 🎯 상대좌표화 ===
    pose = keypoints[0]
    ref_point = pose[0]  # 코를 기준
    keypoints = [kp - ref_point for kp in keypoints]

    # === 📏 스케일 정규화 ===
    all_coords = np.concatenate(keypoints, axis=0)
    std = np.std(all_coords, axis=0) + 1e-6
    keypoints = [(kp) / std for kp in keypoints]

    # flatten
    full_frame = np.concatenate(keypoints, axis=0)  # (137, 2)
    return full_frame.flatten()  # (274,)


# 📁 3. 폴더 구조: NIA_SL_WORD{word}_REAL{person}_{angle}
def collect_all_sample_paths(data_root, sequence_len=50):
    pattern = r'NIA_SL_WORD(\d+)_REAL(\d+)_([A-Z])'
    all_sequences, all_labels = [], []
    label_dict = {}
    label_id = 0

    folder_paths = sorted(glob(os.path.join(data_root, 'NIA_SL_WORD*')))

    for folder in folder_paths:
        folder_name = os.path.basename(folder)
        match = re.match(pattern, folder_name)
        if not match:
            continue
        word_id, person_id, angle = match.groups()
        if word_id not in label_dict:
            label_dict[word_id] = label_id
            label_id += 1

        json_files = sorted(glob(os.path.join(folder, '*.json')))
        if len(json_files) >= sequence_len:
            loaded_seq = [load_keypoint_json(f) for f in json_files]
            interp_seq = interpolate_sequence(np.array(loaded_seq), target_len=120)
            all_sequences.append(interp_seq)
            all_labels.append(label_dict[word_id])

    # label_dict는 {"1506": 0, "1511": 1, ...}
    import json

    id_to_label = {v: WORD_LABELS[k] for k, v in label_dict.items()}
    with open("label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(id_to_label, f, ensure_ascii=False, indent=2)

    # ✅ 사람이 읽을 수 있는 단어명 리스트로 출력
    label_names = [WORD_LABELS[wid] for wid in sorted(label_dict.keys())]
    return np.array(all_sequences), np.array(all_labels), label_names

# 🧺 4. PyTorch Dataset + DataLoader
class SignKeypointDataset(Dataset):
    def __init__(self, sequences, labels):
        self.data = sequences
        self.labels = labels

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data)
    
# 🧠 5. LSTM 분류기 모델 정의
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=134, hidden_dim=HIDDEN_DIM, num_layers=1, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# 🏋️‍♂️ 6. 학습 루프

def train(model, dataloader, optimizer, criterion, device, epochs=20):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct, total = 0, 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (pred.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f} | Acc: {acc:.4f}")

# 📈 7. 평가 함수 + 혼동 행렬 출력
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate(model, dataset, label_names):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    preds, trues = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            pred = out.argmax(dim=1).cpu().item()
            preds.append(pred)
            trues.append(y.item())

    num_classes = len(label_names)
    label_indices = list(range(num_classes))

    # ✅ labels 인자 명시
    print("\n📊 Classification Report:")
    print(classification_report(
        trues, preds,
        labels=label_indices,
        target_names=label_names,
        digits=4,
        zero_division=0  # 0으로 나눌 때 오류 방지
    ))

    # 📉 Confusion Matrix 출력
    cm = confusion_matrix(trues, preds, labels=label_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()


# 🚀 8. 실행 코드
if __name__ == '__main__':
    X, y, label_names = collect_all_sample_paths("data_train")
    dataset = SignKeypointDataset(X, y)
    X_val, y_val, label_names = collect_all_sample_paths("data_val")
    dataset_eval = SignKeypointDataset(X_val, y_val)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = LSTMClassifier(input_dim=134, hidden_dim=HIDDEN_DIM, num_layers=1, num_classes=len(label_names))

    # 🔧 자동 클래스 가중치 계산
    weights = compute_class_weights(y, num_classes=len(label_names))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # le-3 

    # 🚀 학습
    train(model, dataloader, optimizer, criterion, device, epochs=50)
    from collections import Counter
    print("전체 샘플 수:", len(y))
    print("클래스별 개수:", Counter(y))

    # 평가용 데이터셋 안에 클래스별 개수 확인
    evaluate_labels = [label for _, label in DataLoader(dataset, batch_size=1)]
    print("evaluate용 클래스 분포:", Counter([y.item() for y in evaluate_labels]))

    # 💾 저장 및 평가
    torch.save(model.state_dict(), "best_model.pt")
    evaluate(model, dataset_eval, label_names)


