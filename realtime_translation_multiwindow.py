import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import torch
from torch import nn
import numpy as np
from collections import deque, Counter
import mediapipe as mp  
import json

# ⚙️ 설정값
WINDOW_SIZES = list(range(50, 151, 10))  # [50, 60, ..., 150]
FIXED_INPUT_LEN = 120
INPUT_DIM = 134
MODEL_PATH = "best_model.pt"
CONFIDENCE_THRESH = 0.35
HIDDEN_DIM = 20

with open("label_mapping.json", "r", encoding="utf-8") as f:
    LABELS = json.load(f)
LABELS = [LABELS[str(i)] for i in range(len(LABELS))]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=1, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
    
def classify_speed(ws_list):
    speed_counts = {"Fast": 0, "Normal": 0, "Slow": 0}
    for ws in ws_list:
        if ws <= 80:
            speed_counts["Fast"] += 1
        elif ws >= 120:
            speed_counts["Slow"] += 1
        else:
            speed_counts["Normal"] += 1

    final_speed = max(speed_counts.items(), key=lambda x: x[1])[0]

    # 색상 정의
    color_map = {
        "Fast": (0, 255, 255),     # 노란색
        "Normal": (255, 255, 0),   # 하늘색
        "Slow": (255, 100, 100)    # 연한 빨강
    }

    return final_speed, color_map[final_speed]


import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_face_mesh_tesselation_style

def draw_selected_landmarks(image, results, width, height):
    # 🟦 Pose: 25개만
    if results.pose_landmarks:
        for idx in range(25):
            lm = results.pose_landmarks.landmark[idx]
            x, y = int(lm.x * width), int(lm.y * height)
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
    
    # 🟨 Face: 앞에서 70개만
    if results.face_landmarks:
        for idx in range(400):
            lm = results.face_landmarks.landmark[idx]
            x, y = int(lm.x * width), int(lm.y * height)
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
    
    # 🟥 Left Hand: 전체 21개
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

    # 🟩 Right Hand: 전체 21개
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

    return image
    
# 🧠 모델 로드
model = LSTMClassifier(
    input_dim=134,
    hidden_dim=HIDDEN_DIM,
    num_layers=1,
    num_classes=4
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 🔧 보간 함수
def interpolate_sequence(seq, target_len=120):
    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, target_len)
    interpolated = np.zeros((target_len, seq.shape[1]))
    for i in range(seq.shape[1]):
        interpolated[:, i] = np.interp(x_new, x_old, seq[:, i])
    return interpolated

# 🎯 최종 결과 결정 함수
def decide_prediction(results):
    # results: [(label, conf, ws), ...]
    filtered = [(label, conf, ws) for label, conf, ws in results if conf > CONFIDENCE_THRESH]
    if not filtered:
        return None, []

    max_conf = max(conf for _, conf, _ in filtered)

    # conf가 max_conf와 동일한 모든 항목 추출
    top_results = [(label, ws) for label, conf, ws in filtered if conf == max_conf]
    
    # 다수결로 label 결정
    labels = [label for label, _ in top_results]
    most_common_label = Counter(labels).most_common(1)[0][0]

    # 강조 대상이 될 ws 목록 반환
    top_ws = [ws for _, ws in top_results]

    return most_common_label, top_ws



# 🌀 MediaPipe 설정
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# 🧺 버퍼 생성
buffers = {ws: deque(maxlen=ws) for ws in WINDOW_SIZES}

# 🎥 Webcam 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
panel_width = 500
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter("output.mp4", fourcc, 30.0, (frame_width + 500, frame_height))  # 오른쪽 패널 포함

# 키포인트 추출 함수
def extract_keypoints(results, width, height):
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark[:25]:
            x = int(lm.x * width)
            y = int(lm.y * height)
            keypoints.extend([x, y])
    else:
        keypoints.extend([0] * 25 * 2)

    # face keypoint는 제외
    '''
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark[0:70]:
            x = int(lm.x * width)
            y = int(lm.y * height)
            keypoints.extend([x, y])
    else:
        keypoints.extend([0] * 70 * 2)
    '''
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            x = int(lm.x * width)
            y = int(lm.y * height)
            keypoints.extend([x, y])
    else:
        keypoints.extend([0] * 21 * 2)

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            x = int(lm.x * width)
            y = int(lm.y * height)
            keypoints.extend([x, y])
    else:
        keypoints.extend([0] * 21 * 2)

    # extract_keypoints() 결과 반환 전에 아래 적용
    keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 2)  # (N, 2)

    ref_point = keypoints[0]  # 코 위치 (x, y)
    keypoints = keypoints - ref_point  # ✅ 모든 점을 상대좌표로

    std = np.std(keypoints, axis=0) + 1e-6
    keypoints = keypoints / std  # ✅ 스케일 정규화

    return keypoints.flatten()



frame_counter = 0  # 프레임 수를 세기 위한 변수
final_pred, highlight_ws_list = None, []

# 🔁 실시간 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1  # 프레임을 1장 읽을 때마다 증가
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    h, w, _ = frame.shape
    keypoints = extract_keypoints(results, w, h)

    results_list = []
    window_outputs = []

    for ws, buffer in buffers.items():
        buffer.append(keypoints)
        label = "-"
        conf = 0.0

        if len(buffer) == ws:
            seq = np.array(buffer)
            interp_seq = interpolate_sequence(seq, target_len=FIXED_INPUT_LEN)
            x = torch.tensor(interp_seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
                prob = torch.softmax(out, dim=1)
                pred = prob.argmax(dim=1).item()
                conf = prob.max().item()
                label = LABELS[pred]
                results_list.append((pred, conf, ws))

        window_outputs.append((ws, label, conf))


    # 5 프레임마다 결과 업데이트
    if frame_counter % 5 == 0:
        final_pred, highlight_ws_list = decide_prediction(results_list)
    '''
    # frame은 BGR 이미지 (cv2.imread or webcam 프레임)
    h, w, _ = frame.shape
    frame = draw_selected_landmarks(frame, results, w, h)
    '''
    # 1️⃣ Keypoint 시각화
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # 2️⃣ 오른쪽 패널 만들기
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    frame = np.hstack((frame, panel))
    
    # 4️⃣ 각 윈도우별 결과 출력
    for i, (ws, label, conf) in enumerate(window_outputs):
        text = f"[{ws} Frames] {label} ({conf:.2f})"

        if ws in highlight_ws_list:
            color = (0, 255, 0)  # 초록색 강조
            thickness = 3
        else:
            color = (200, 200, 200)
            thickness = 2

        cv2.putText(frame, text, (frame_width + 20, 60 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)
        
    # 3️⃣ 최종 예측 출력
    if final_pred is not None:
        label_text = f"Predict: {LABELS[final_pred]}"
        cv2.putText(frame, label_text, (frame_width + 20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        speed_text, speed_color = classify_speed(highlight_ws_list)
        cv2.putText(frame, f"Speed: {speed_text}", (frame_width + 20, 510),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, speed_color, 2)
    else:
        cv2.putText(frame, "No output", (frame_width + 20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    
    
    # 5️⃣ 프레임 출력 및 저장
    cv2.imshow("Multi window SLT", frame)
    out_video.write(frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 마무리
cap.release()
out_video.release()
cv2.destroyAllWindows()
