"""
실시간 지문자 인식 (웹캠)
- V2 모델 사용 (기존 + JSON keypoints 데이터 학습)
- 자동 인식 (사용자 조작 불필요)
- Top 3 예측 결과 + 확률 표시
- 한글 폰트 지원
"""



### 환경 설정

'''
bash
# 1. 터미널에서 conda 가상환경 활성화
conda activate py311_env

# 2. 필요한 패키지 확인 (이미 설치되어 있어야 함)
pip list | grep -E "torch|mediapipe|opencv"
```

### 모델 학습

bash
# 1. 가상환경 활성화
#conda activate py311_env

# 2. 인식 코드가 있는  디렉토리로 이동
#cd src/common/recognition

# 3. 학습 스크립트 실행
#python realtime_recognition.py

'''

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import warnings

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
warnings.filterwarnings('ignore')

# ==================== 설정 ====================
MODEL_PATH = str(_ROOT / "models/photo_model/photo_finger_alphabet_411_model_V2.pt")

# 키포인트 차원
POSE_DIM = 75
FACE_DIM = 210
HAND_DIM = 63
TOTAL_DIM = 411

# 가중치
POSE_WEIGHT = 0.005
FACE_WEIGHT = 0.005
HAND_WEIGHT = 0.99

# 디바이스
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"디바이스: {device}")

# ==================== 한글 폰트 설정 ====================
def get_korean_font(size=30):
    """시스템에서 한글 폰트 찾기"""
    font_paths = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()

font_large = get_korean_font(40)
font_medium = get_korean_font(30)
font_small = get_korean_font(20)

def put_korean_text(img, text, pos, font, color=(255, 255, 255)):
    """OpenCV 이미지에 한글 텍스트 그리기"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==================== 모델 정의 ====================
class KeypointClassifier(nn.Module):
    def __init__(self, input_size=411, num_classes=28, dropout=0.3):
        super(KeypointClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ==================== 모델 로드 ====================
print("모델 로드 중...")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

num_classes = checkpoint['num_classes']
idx_to_label = checkpoint['idx_to_label']
norm_mean = checkpoint['norm_params']['mean'].to(device)
norm_std = checkpoint['norm_params']['std'].to(device)

model = KeypointClassifier(input_size=TOTAL_DIM, num_classes=num_classes, dropout=0.0)
model.load_state_dict(checkpoint['model_kp_state'])
model = model.to(device)
model.eval()

print(f"클래스 수: {num_classes}")
print(f"라벨: {list(idx_to_label.values())}")

# ==================== MediaPipe 초기화 ====================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ==================== 키포인트 추출 함수 ====================
def extract_keypoints(image, pose, hands, face_mesh):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_image)
    hands_results = hands.process(rgb_image)
    face_results = face_mesh.process(rgb_image)

    detection_info = {'pose': False, 'face': False, 'left_hand': False, 'right_hand': False}

    # Pose (75)
    pose_kps = []
    if pose_results.pose_landmarks:
        detection_info['pose'] = True
        for i in range(25):
            if i < len(pose_results.pose_landmarks.landmark):
                lm = pose_results.pose_landmarks.landmark[i]
                pose_kps.extend([lm.x, lm.y, lm.visibility])
            else:
                pose_kps.extend([0.0, 0.0, 0.0])
    else:
        pose_kps = [0.0] * POSE_DIM

    # Face (210)
    face_kps = []
    if face_results.multi_face_landmarks:
        detection_info['face'] = True
        face_landmarks = face_results.multi_face_landmarks[0]
        for i in range(70):
            if i < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[i]
                face_kps.extend([lm.x, lm.y, lm.z])
            else:
                face_kps.extend([0.0, 0.0, 0.0])
    else:
        face_kps = [0.0] * FACE_DIM

    # Hands (126)
    left_hand_kps = [0.0] * HAND_DIM
    right_hand_kps = [0.0] * HAND_DIM

    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            handedness = hands_results.multi_handedness[idx].classification[0].label
            hand_kps = []
            for lm in hand_landmarks.landmark:
                hand_kps.extend([lm.x, lm.y, lm.z])

            if handedness == "Left":
                left_hand_kps = hand_kps
                detection_info['left_hand'] = True
            elif handedness == "Right":
                right_hand_kps = hand_kps
                detection_info['right_hand'] = True

    combined = np.concatenate([
        np.array(pose_kps),
        np.array(face_kps),
        np.array(left_hand_kps),
        np.array(right_hand_kps)
    ])

    return combined.astype(np.float32), detection_info, hands_results

def apply_weights(keypoints):
    weighted = keypoints.copy()
    weighted[0:POSE_DIM] *= POSE_WEIGHT
    weighted[POSE_DIM:POSE_DIM+FACE_DIM] *= FACE_WEIGHT
    weighted[POSE_DIM+FACE_DIM:POSE_DIM+FACE_DIM+HAND_DIM] *= HAND_WEIGHT
    weighted[POSE_DIM+FACE_DIM+HAND_DIM:TOTAL_DIM] *= HAND_WEIGHT
    return weighted

# ==================== 예측 함수 ====================
def predict_top3(keypoints):
    weighted = apply_weights(keypoints)
    x = torch.tensor(weighted, dtype=torch.float32).unsqueeze(0).to(device)
    x = (x - norm_mean) / norm_std

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        top3_probs, top3_indices = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = top3_indices[0][i].item()
        prob = top3_probs[0][i].item() * 100
        label = idx_to_label[idx]
        results.append((label, prob))

    return results

# ==================== 메인 루프 ====================
print("\n" + "="*50)
print("실시간 지문자 인식 시작")
print("종료: 'q' 키")
print("="*50 + "\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose, \
     mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5) as hands, \
     mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.flip(frame, 1)  # 거울 모드 끔 - 오른손 인식을 위해

        # 키포인트 추출 및 예측
        keypoints, detection_info, hands_results = extract_keypoints(frame, pose, hands, face_mesh)

        # 손 랜드마크 그리기
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

        # 손이 감지되면 예측
        hand_detected = detection_info['left_hand'] or detection_info['right_hand']

        # 배경 박스
        cv2.rectangle(frame, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 200), (255, 255, 255), 2)

        if hand_detected:
            top3 = predict_top3(keypoints)

            # 한글 텍스트 표시
            frame = put_korean_text(frame, "[ TOP 3 예측 결과 ]", (20, 20), font_medium, (0, 255, 255))

            colors = [(0, 255, 0), (255, 200, 0), (255, 150, 0)]
            for i, (label, prob) in enumerate(top3):
                rank = i + 1
                text = f"{rank}위: {label}  ({prob:.1f}%)"
                y_pos = 70 + i * 40
                frame = put_korean_text(frame, text, (20, y_pos), font_medium, colors[i])
        else:
            frame = put_korean_text(frame, "[ 손을 기다리는 중... ]", (20, 20), font_medium, (0, 255, 255))
            frame = put_korean_text(frame, "손을 카메라에 보여주세요", (20, 70), font_small, (200, 200, 200))

        # 상태 표시
        status = f"손 감지: {'O' if hand_detected else 'X'}"
        frame = put_korean_text(frame, status, (20, 170), font_small, (150, 150, 150))

        cv2.imshow('Finger Alphabet Recognition (Press Q to quit)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\n종료되었습니다.")
