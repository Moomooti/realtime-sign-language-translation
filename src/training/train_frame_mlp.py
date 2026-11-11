# =============================================================================
# train_frame_mlp.py
# =============================================================================

import os
import re
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from collections import Counter

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────
DATA_DIR  = str(_ROOT / "dataset/자체제작_단어_keypoints")
SAVE_PATH = str(_ROOT / "models/video_model/frame_mlp.pt")

BATCH_SIZE           = 128
NUM_EPOCHS           = 80
LR                   = 0.001
PATIENCE             = 15
MAX_FRAMES_PER_CLASS = 500
# ─────────────────────────────────────────────────────────


# =============================================================================
# 1. 모델 (3레이어, 512 hidden)
# =============================================================================
class FrameMLP(nn.Module):
    def __init__(self, input_dim=114, hidden_dim=512, num_classes=271):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),  
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 2. 특징 추출
# =============================================================================
def normalize_hand(raw, n=63):
    """손목 기준 상대좌표 -> (21,2) ndarray, has_hand"""
    if not raw:
        return np.zeros((21, 2), dtype=np.float32), False

    arr = np.array(raw[:n], dtype=np.float32)
    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n - len(arr), dtype=np.float32)])

    pts      = arr.reshape(-1, 3)[:, :2]
    has_hand = (arr[:2].max() > 0.01)

    wrist = pts[0].copy()
    pts   = pts - wrist
    scale = np.linalg.norm(pts[9]) + 1e-8
    pts   = pts / scale

    return pts.astype(np.float32), has_hand


def compute_finger_angles(pts_21x2):
    """손가락 관절 각도(코사인) -> 15차원"""
    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]
    angles = []
    for finger in fingers:
        for i in range(len(finger) - 2):
            a  = pts_21x2[finger[i]]
            b  = pts_21x2[finger[i + 1]]
            c  = pts_21x2[finger[i + 2]]
            ba = a - b
            bc = c - b
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angles.append(float(np.clip(cos, -1, 1)))
    return np.array(angles, dtype=np.float32)


def extract_hand_features(pts_21x2, has_hand):
    """상대좌표 42 + 각도 15 = 57차원. 손 없으면 0벡터."""
    if not has_hand:
        return np.zeros(57, dtype=np.float32)
    return np.concatenate([pts_21x2.flatten(), compute_finger_angles(pts_21x2)])


def determine_dominant_field(folder_path):
    """
    폴더 내 모든 JSON을 스캔해 손목 누적 이동량을 비교 →
    더 많이 움직인 손 JSON 필드명 반환: 'hand_left_keypoints_2d' 또는 'hand_right_keypoints_2d'

    규칙:
      - 이동량 차이가 1.2배 이상 → 많이 움직인 쪽
      - 비슷하면 → 더 많은 프레임에 등장한 쪽
      - 둘 다 없으면 → 'hand_left_keypoints_2d' (fallback)
    """
    jsons = sorted(glob.glob(os.path.join(folder_path, '*.json')))

    left_wrists  = []
    right_wrists = []

    for jp in jsons:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'people' not in data or not data['people']:
                continue
            p = data['people'][0]

            raw_l = p.get('hand_left_keypoints_2d',  [])
            raw_r = p.get('hand_right_keypoints_2d', [])

            # 손목(인덱스 0) x, y 유효성: x 또는 y > 0.01
            if len(raw_l) >= 2 and max(raw_l[0], raw_l[1]) > 0.01:
                left_wrists.append((raw_l[0], raw_l[1]))
            if len(raw_r) >= 2 and max(raw_r[0], raw_r[1]) > 0.01:
                right_wrists.append((raw_r[0], raw_r[1]))
        except Exception:
            continue

    def _total_motion(wrists):
        total = 0.0
        for i in range(1, len(wrists)):
            dx = wrists[i][0] - wrists[i-1][0]
            dy = wrists[i][1] - wrists[i-1][1]
            total += (dx**2 + dy**2) ** 0.5
        return total

    lm = _total_motion(left_wrists)
    rm = _total_motion(right_wrists)

    if rm > lm * 1.2:
        return 'hand_right_keypoints_2d'
    elif lm > rm * 1.2:
        return 'hand_left_keypoints_2d'
    else:
        # 이동량 비슷 → 등장 프레임 수로 결정
        return ('hand_left_keypoints_2d' if len(left_wrists) >= len(right_wrists)
                else 'hand_right_keypoints_2d')


def load_hand_keypoints(json_path, dominant_field):
    """
    JSON 1프레임 → 114차원 [dominant(0-56) | support(57-113)]
    dominant_field: 'hand_left_keypoints_2d' 또는 'hand_right_keypoints_2d'
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'people' not in data or not data['people']:
            return None
        p = data['people'][0]
    except Exception:
        return None

    support_field = ('hand_right_keypoints_2d'
                     if dominant_field == 'hand_left_keypoints_2d'
                     else 'hand_left_keypoints_2d')

    raw_dom = p.get(dominant_field,  [])
    raw_sup = p.get(support_field,   [])

    pts_dom, has_dom = normalize_hand(raw_dom)
    pts_sup, has_sup = normalize_hand(raw_sup)

    if not has_dom and not has_sup:
        return None

    feat_dom = extract_hand_features(pts_dom, has_dom)
    feat_sup = extract_hand_features(pts_sup, has_sup)
    return np.concatenate([feat_dom, feat_sup])


def extract_label(folder_name):
    name = re.sub(r'\(\d+\)$', '', folder_name)
    name = re.sub(r'_\d+$', '', name)
    return name


# =============================================================================
# 3. 데이터셋 로드
# =============================================================================
def load_dataset(data_dir):
    folders = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])

    unique_labels = sorted(set(extract_label(f) for f in folders))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label  = {i: l for l, i in label_to_idx.items()}

    print(f"폴더 수: {len(folders)} | 클래스 수: {len(unique_labels)}")

    class_samples = {i: [] for i in range(len(unique_labels))}
    skipped = 0

    for folder in folders:
        label        = extract_label(folder)
        idx          = label_to_idx[label]
        folder_path  = os.path.join(data_dir, folder)

        # 폴더 단위로 dominant hand 판별 (한 번만)
        dominant_field = determine_dominant_field(folder_path)

        jsons = sorted(glob.glob(os.path.join(folder_path, '*.json')))
        for jp in jsons:
            kp = load_hand_keypoints(jp, dominant_field)
            if kp is None:
                skipped += 1
                continue
            class_samples[idx].append(kp)

    counts = {idx_to_label[i]: len(v) for i, v in class_samples.items() if v}
    print(f"\n  [진단] 손 없는 프레임: {skipped}개 제외")
    print(f"  [진단] 클래스별 프레임 수 (상위 10개):")
    for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        bar = '#' * min(cnt // 10, 30)
        print(f"    {lbl:12s}: {cnt:5d} {bar}")
    print(f"  [진단] 총 프레임: {sum(counts.values())}개")
    print(f"  [진단] 클래스당 상한: {MAX_FRAMES_PER_CLASS}프레임\n")

    X, Y = [], []
    rng  = np.random.default_rng(42)
    for idx, samples in class_samples.items():
        if not samples:
            continue
        if len(samples) > MAX_FRAMES_PER_CLASS:
            chosen  = rng.choice(len(samples), MAX_FRAMES_PER_CLASS, replace=False)
            samples = [samples[i] for i in chosen]
        X.extend(samples)
        Y.extend([idx] * len(samples))

    print(f"  최종 샘플: {len(X)}개 (상한 적용 후)")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64), label_to_idx, idx_to_label


# =============================================================================
# 4. Dataset
# =============================================================================
class FrameDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# =============================================================================
# 5. 학습
# =============================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}\n")

    X, Y, label_to_idx, idx_to_label = load_dataset(DATA_DIR)
    num_classes = len(label_to_idx)
    input_dim   = X.shape[1]

    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X_n  = (X - mean) / std

    N   = len(X_n)
    idx = np.random.permutation(N)
    tr_end, va_end = int(0.8 * N), int(0.9 * N)
    tr, va, te     = idx[:tr_end], idx[tr_end:va_end], idx[va_end:]

    class_counts   = Counter(Y[tr].tolist())
    sample_weights = [1.0 / class_counts[int(y)] for y in Y[tr]]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    tr_ld = DataLoader(FrameDataset(X_n[tr], Y[tr]), BATCH_SIZE, sampler=sampler)
    va_ld = DataLoader(FrameDataset(X_n[va], Y[va]), BATCH_SIZE)
    te_ld = DataLoader(FrameDataset(X_n[te], Y[te]), BATCH_SIZE)

    model = FrameMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
    crit  = nn.CrossEntropyLoss()

    best_val, best_weights, patience_cnt = float('inf'), None, 0

    print("=" * 55)
    print(f"학습 시작 | input={input_dim}d | epochs={NUM_EPOCHS} | batch={BATCH_SIZE}")
    print("=" * 55)

    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        tr_loss = tr_cor = tr_tot = 0
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out  = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(bx)
            tr_cor  += (out.argmax(1) == by).sum().item()
            tr_tot  += len(bx)

        model.eval()
        va_loss = va_cor = va_tot = 0
        with torch.no_grad():
            for bx, by in va_ld:
                bx, by = bx.to(device), by.to(device)
                out     = model(bx)
                loss    = crit(out, by)
                va_loss += loss.item() * len(bx)
                va_cor  += (out.argmax(1) == by).sum().item()
                va_tot  += len(bx)

        tr_acc = 100 * tr_cor / tr_tot
        va_acc = 100 * va_cor / va_tot
        va_l   = va_loss / va_tot
        sched.step(va_l)

        print(f"  Ep {ep:3d} | tr_acc={tr_acc:.1f}% | va_acc={va_acc:.1f}% | va_loss={va_l:.4f}")

        if va_l < best_val - 1e-4:
            best_val     = va_l
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stop at ep {ep}")
                break

    if best_weights:
        model.load_state_dict(best_weights)

    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for bx, by in te_ld:
            all_p.extend(model(bx.to(device)).argmax(1).cpu().numpy())
            all_y.extend(by.numpy())
    acc = 100 * np.mean(np.array(all_p) == np.array(all_y))
    f1  = f1_score(all_y, all_p, average='weighted', zero_division=0) * 100
    print(f"\n  테스트 결과 | Acc={acc:.2f}% | F1={f1:.2f}%")

    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params':      {'mean': mean.tolist(), 'std': std.tolist()},
        'label_to_idx':     label_to_idx,
        'idx_to_label':     idx_to_label,
        'input_dim':        input_dim,
        'num_classes':      num_classes,
    }, SAVE_PATH)
    print(f"\n  저장 완료: {SAVE_PATH}")
    print(f"  입력 차원: {input_dim} (상대좌표 84 + 각도 30)")


if __name__ == "__main__":
    train()
