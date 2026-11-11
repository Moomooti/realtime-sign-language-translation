"""
사진 지문자 인식 모델 학습 스크립트 (키포인트 모델)
- 411차원 키포인트 (Pose + Face + Hands)
- 손에 0.99 가중치 적용
- 기존 이미지 데이터 + JSON keypoints 데이터 병합 학습
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

# 2. 모델 디렉토리로 이동
#cd src/training/photo_model

# 3. 학습 스크립트 실행
#python run_training.py

'''



import os
import cv2
import json
import numpy as np
import mediapipe as mp
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import glob
import warnings

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
warnings.filterwarnings('ignore')

# ==================== 설정 ====================
DATA_DIR = str(_ROOT / "raw/photo_alphabet")  # 기존 이미지 데이터
KEYPOINTS_DIR = str(_ROOT / "dataset/자체제작_사진_지문자_keypoints")  # 새 JSON keypoints 데이터
MODEL_DIR = str(_ROOT / "models/photo_model")

# 키포인트 차원 설정
POSE_DIM = 75      # 25 landmarks * 3
FACE_DIM = 210     # 70 landmarks * 3
HAND_DIM = 63      # 21 landmarks * 3
TOTAL_DIM = POSE_DIM + FACE_DIM + HAND_DIM * 2  # 411

# 부위별 가중치
POSE_WEIGHT = 0.005
FACE_WEIGHT = 0.005
HAND_WEIGHT = 0.99

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"사용 디바이스: {device}")
print(f"키포인트 차원: {TOTAL_DIM}")
print(f"손 가중치: {HAND_WEIGHT}")

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# ==================== 데이터 로드 ====================
print("\n" + "="*50)
print("1. 데이터 로드")
print("="*50)

# 1-1. 기존 이미지 데이터
image_files = glob.glob(os.path.join(DATA_DIR, "*.png"))
image_files.extend(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
image_files.extend(glob.glob(os.path.join(DATA_DIR, "*.jpeg")))
image_files = sorted(image_files)

print(f"기존 이미지 데이터: {len(image_files)}개")

# 1-2. JSON keypoints 데이터 (각도별 1, 2, 3 폴더)
json_keypoint_files = []
for angle_folder in ['1', '2', '3']:
    angle_path = os.path.join(KEYPOINTS_DIR, angle_folder)
    if os.path.exists(angle_path):
        for label_folder in os.listdir(angle_path):
            label_path = os.path.join(angle_path, label_folder)
            if os.path.isdir(label_path):
                for json_file in glob.glob(os.path.join(label_path, "*_keypoints.json")):
                    json_keypoint_files.append({
                        'path': json_file,
                        'label': label_folder,
                        'angle': angle_folder
                    })

print(f"JSON keypoints 데이터: {len(json_keypoint_files)}개 (3개 각도)")

# 라벨 추출 (기존 이미지 + JSON 데이터 모두 포함)
labels = []
for img_path in image_files:
    filename = os.path.basename(img_path)
    label = os.path.splitext(filename)[0]
    labels.append(label)

for json_info in json_keypoint_files:
    labels.append(json_info['label'])

unique_labels = sorted(set(labels))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(unique_labels)

print(f"클래스 수: {num_classes}")
print(f"라벨: {unique_labels}")

# ==================== 키포인트 추출 함수 ====================
def extract_full_keypoints(image, pose, hands, face_mesh):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_image)
    hands_results = hands.process(rgb_image)
    face_results = face_mesh.process(rgb_image)

    detection_info = {'pose': False, 'face': False, 'left_hand': False, 'right_hand': False}

    # Pose keypoints (75)
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

    # Face keypoints (210)
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

    # Hand keypoints (126)
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

    combined_kps = np.concatenate([
        np.array(pose_kps),
        np.array(face_kps),
        np.array(left_hand_kps),
        np.array(right_hand_kps)
    ])

    return combined_kps.astype(np.float32), detection_info

def load_json_keypoints(json_path):
    """JSON 파일에서 키포인트 로드 (411차원)"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data.get('people') or len(data['people']) == 0:
        return None

    person = data['people'][0]

    pose_kps = person.get('pose_keypoints_2d', [0.0] * POSE_DIM)
    face_kps = person.get('face_keypoints_2d', [0.0] * FACE_DIM)
    left_hand_kps = person.get('hand_left_keypoints_2d', [0.0] * HAND_DIM)
    right_hand_kps = person.get('hand_right_keypoints_2d', [0.0] * HAND_DIM)

    # 길이 맞추기
    pose_kps = (pose_kps + [0.0] * POSE_DIM)[:POSE_DIM]
    face_kps = (face_kps + [0.0] * FACE_DIM)[:FACE_DIM]
    left_hand_kps = (left_hand_kps + [0.0] * HAND_DIM)[:HAND_DIM]
    right_hand_kps = (right_hand_kps + [0.0] * HAND_DIM)[:HAND_DIM]

    combined_kps = np.concatenate([
        np.array(pose_kps),
        np.array(face_kps),
        np.array(left_hand_kps),
        np.array(right_hand_kps)
    ])

    return combined_kps.astype(np.float32)

# ==================== 키포인트 추출 ====================
print("\n" + "="*50)
print("2. 키포인트 추출")
print("="*50)

all_keypoints = []
all_labels_idx = []

# 2-1. 기존 이미지에서 키포인트 추출
print("\n[기존 이미지 데이터 처리]")
with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose, \
     mp_hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5) as hands, \
     mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

    for idx, img_path in enumerate(image_files):
        image = cv2.imread(img_path)
        if image is None:
            continue

        keypoints, detection_info = extract_full_keypoints(image, pose, hands, face_mesh)

        filename = os.path.basename(img_path)
        label = os.path.splitext(filename)[0]
        label_idx = label_to_idx[label]

        all_keypoints.append(keypoints)
        all_labels_idx.append(label_idx)

        hand_detected = detection_info['left_hand'] or detection_info['right_hand']
        print(f"  [{idx+1}/{len(image_files)}] {label} - Pose:{detection_info['pose']} Face:{detection_info['face']} Hand:{hand_detected}")

print(f"\n기존 이미지 키포인트: {len(all_keypoints)}개")

# 2-2. JSON keypoints 데이터 로드
print("\n[JSON keypoints 데이터 처리]")
json_loaded = 0
json_failed = 0

for idx, json_info in enumerate(json_keypoint_files):
    keypoints = load_json_keypoints(json_info['path'])

    if keypoints is not None:
        label_idx = label_to_idx[json_info['label']]
        all_keypoints.append(keypoints)
        all_labels_idx.append(label_idx)
        json_loaded += 1
    else:
        json_failed += 1

    if (idx + 1) % 20 == 0 or idx == len(json_keypoint_files) - 1:
        print(f"  [{idx+1}/{len(json_keypoint_files)}] JSON 로드 완료")

print(f"\nJSON 키포인트: {json_loaded}개 로드 성공, {json_failed}개 실패")

all_keypoints = np.array(all_keypoints, dtype=np.float32)
print(f"\n전체 키포인트 shape: {all_keypoints.shape}")

# ==================== 가중치 적용 및 데이터 증강 ====================
print("\n" + "="*50)
print("3. 가중치 적용 및 데이터 증강")
print("="*50)

def apply_body_part_weights(keypoints):
    weighted_kp = keypoints.copy()
    weighted_kp[0:POSE_DIM] *= POSE_WEIGHT
    weighted_kp[POSE_DIM:POSE_DIM+FACE_DIM] *= FACE_WEIGHT
    weighted_kp[POSE_DIM+FACE_DIM:POSE_DIM+FACE_DIM+HAND_DIM] *= HAND_WEIGHT
    weighted_kp[POSE_DIM+FACE_DIM+HAND_DIM:TOTAL_DIM] *= HAND_WEIGHT
    return weighted_kp

def augment_keypoints(keypoints, num_augment=30):
    augmented = [keypoints.copy()]

    for _ in range(num_augment):
        kp = keypoints.copy()

        # 노이즈
        noise = np.random.normal(0, 0.015, kp.shape)
        kp += noise

        # 스케일
        scale = np.random.uniform(0.95, 1.05)
        for i in range(0, len(kp), 3):
            if i + 1 < len(kp):
                kp[i] = (kp[i] - 0.5) * scale + 0.5
                kp[i+1] = (kp[i+1] - 0.5) * scale + 0.5

        # 회전
        angle = np.random.uniform(-10, 10) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for i in range(0, len(kp), 3):
            if i + 1 < len(kp):
                x, y = kp[i] - 0.5, kp[i+1] - 0.5
                kp[i] = x * cos_a - y * sin_a + 0.5
                kp[i+1] = x * sin_a + y * cos_a + 0.5

        augmented.append(kp)

    return np.array(augmented, dtype=np.float32)

augmented_keypoints = []
augmented_labels = []

for i in range(len(all_keypoints)):
    kp = all_keypoints[i]
    weighted_kp = apply_body_part_weights(kp)
    aug_kps = augment_keypoints(weighted_kp, 30)

    for aug_kp in aug_kps:
        augmented_keypoints.append(aug_kp)
        augmented_labels.append(all_labels_idx[i])

X_keypoints_aug = torch.tensor(np.array(augmented_keypoints), dtype=torch.float32)
Y_labels_aug = torch.tensor(augmented_labels, dtype=torch.long)

print(f"증강 전: {len(all_keypoints)}개")
print(f"증강 후: {len(augmented_keypoints)}개")

# 정규화
kp_mean = X_keypoints_aug.mean(dim=0, keepdim=True)
kp_std = X_keypoints_aug.std(dim=0, keepdim=True) + 1e-8
X_keypoints_norm = (X_keypoints_aug - kp_mean) / kp_std

norm_params = {'mean': kp_mean, 'std': kp_std}

# ==================== 키포인트 모델 정의 ====================
print("\n" + "="*50)
print("4. 키포인트 모델 학습")
print("="*50)

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

model_kp = KeypointClassifier(input_size=TOTAL_DIM, num_classes=num_classes, dropout=0.3)
model_kp = model_kp.to(device)

# 데이터셋
dataset_kp = TensorDataset(X_keypoints_norm, Y_labels_aug)
train_size = int(0.8 * len(dataset_kp))
val_size = len(dataset_kp) - train_size
train_dataset_kp, val_dataset_kp = random_split(dataset_kp, [train_size, val_size])

train_loader_kp = DataLoader(train_dataset_kp, batch_size=32, shuffle=True)
val_loader_kp = DataLoader(val_dataset_kp, batch_size=32, shuffle=False)

print(f"학습: {len(train_dataset_kp)}개, 검증: {len(val_dataset_kp)}개")

# 학습
criterion = nn.CrossEntropyLoss()
optimizer_kp = optim.Adam(model_kp.parameters(), lr=0.001, weight_decay=1e-4)
scheduler_kp = optim.lr_scheduler.ReduceLROnPlateau(optimizer_kp, mode='min', factor=0.5, patience=10)

num_epochs = 100
best_val_acc = 0

for epoch in range(num_epochs):
    model_kp.train()
    train_loss = 0
    train_correct = 0

    for batch_x, batch_y in train_loader_kp:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer_kp.zero_grad()
        outputs = model_kp(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_kp.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == batch_y).sum().item()

    train_loss /= len(train_loader_kp)
    train_acc = 100 * train_correct / len(train_dataset_kp)

    model_kp.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader_kp:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model_kp(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_y).sum().item()

    val_loss /= len(val_loader_kp)
    val_acc = 100 * val_correct / len(val_dataset_kp)

    scheduler_kp.step(val_loss)

    if (epoch + 1) % 10 == 0 or val_acc > best_val_acc:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {train_loss:.4f}/{val_loss:.4f} | Acc: {train_acc:.1f}%/{val_acc:.1f}%", end="")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model_kp.state_dict(), os.path.join(MODEL_DIR, 'model_keypoint_411_best_V2.pth'))
            print(" <- BEST")
        else:
            print()

print(f"\n키포인트 모델 Best Accuracy: {best_val_acc:.2f}%")

# ==================== 모델 저장 ====================
print("\n" + "="*50)
print("5. 모델 저장")
print("="*50)

save_data = {
    'model_kp_state': model_kp.state_dict(),
    'norm_params': norm_params,
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'num_classes': num_classes,
    'kp_input_size': TOTAL_DIM,
    'keypoint_structure': {
        'pose_dim': POSE_DIM,
        'face_dim': FACE_DIM,
        'hand_dim': HAND_DIM,
        'total_dim': TOTAL_DIM
    },
    'body_part_weights': {
        'pose': POSE_WEIGHT,
        'face': FACE_WEIGHT,
        'hand': HAND_WEIGHT
    }
}

torch.save(save_data, os.path.join(MODEL_DIR, 'photo_finger_alphabet_411_model_V2.pt'))

print(f"모델 저장 완료: {MODEL_DIR}/photo_finger_alphabet_411_model_V2.pt")
print(f"\n=== 학습 완료 ===")
print(f"키포인트 모델: {best_val_acc:.2f}%")
print(f"\n손 가중치 {HAND_WEIGHT} 적용됨!")
