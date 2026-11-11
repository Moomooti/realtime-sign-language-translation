# =============================================================================
# realtime_dual_stream.py
#
# Primary  (Stream A): Frame MLP  → 매 프레임 손 키포인트 → 세그먼트 다수결
# Secondary(Stream B): CNN-GRU    → 세그먼트 종료 시 확인·신뢰도 보정 (선택)
#
# CNN-GRU 모델 파일이 없으면 MLP 단독으로 동작합니다.
# =============================================================================

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import time
from collections import deque, Counter

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────
MLP_MODEL_PATH       = str(_ROOT / "models/video_model/frame_mlp.pt")
CNN_MODEL_PATH       = str(_ROOT / "models/video_model/sign_language_cnn_gru.pt")   # 없으면 스킵

CONFIDENCE_THRESHOLD = 0.45   # 예측 표시 임계값
MOTION_THRESHOLD     = 0.008  # 손 모션 감지 민감도 (느린 동작 대응)
PRED_COOLDOWN_SEC    = 1.2    # 예측 커밋 최소 간격(초)
WORD_HISTORY_SIZE    = 6      # 단어 히스토리 크기
MLP_VOTE_WINDOW      = 20     # 라이브 다수결 윈도우(프레임)
CNN_CONFIRM_BOOST    = 1.15   # CNN이 동의할 때 신뢰도 배수
CNN_OVERRIDE_CONF    = 0.70   # CNN이 이 이상 확신하면 MLP override 가능
MLP_LOW_CONF         = 0.55   # MLP가 이 이하면 불확실로 간주
# ─────────────────────────────────────────────────────────


# =============================================================================
# 1. 모델 정의
# =============================================================================
class FrameMLP(nn.Module):
    def __init__(self, input_dim=114, num_classes=271):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CnnGruModel(nn.Module):
    def __init__(self, input_dim=411, hidden_dim=128, num_layers=2,
                 num_classes=10, dropout=0.3):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1), nn.BatchNorm1d(256), nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, hidden_dim, 1), nn.BatchNorm1d(hidden_dim), nn.ELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ELU(),
        )
        gru_drop = dropout if num_layers > 1 else 0
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=gru_drop, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.spatial(x)
        x = self.temporal(x)
        x = x.permute(0, 2, 1)
        o, _ = self.gru(x)
        a = self.attention(o)
        c = torch.sum(o * a, dim=1)
        return self.classifier(c)


# =============================================================================
# 2. 모델 로드
# =============================================================================
def load_mlp_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(path, map_location=device, weights_only=False)

    model = FrameMLP(input_dim=ckpt.get('input_dim', 114),
                     num_classes=ckpt['num_classes']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    mean = np.array(ckpt['norm_params']['mean']).flatten().astype(np.float32)
    std  = np.array(ckpt['norm_params']['std']).flatten().astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std)

    print(f"  ✓ MLP (Primary) | 클래스={ckpt['num_classes']} | 입력={ckpt.get('input_dim',114)}d")
    return model, {'mean': mean, 'std': std}, ckpt['idx_to_label'], device


def load_cnn_model(path, device):
    if not os.path.exists(path):
        print(f"  ℹ CNN-GRU 없음 → MLP 단독 동작")
        return None, None, None, None

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt['config']

    model = CnnGruModel(
        input_dim=cfg['input_dim'], hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'], num_classes=ckpt['num_classes'], dropout=0
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    mean = np.array(ckpt['norm_params']['mean']).flatten().astype(np.float32)
    std  = np.array(ckpt['norm_params']['std']).flatten().astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std)

    print(f"  ✓ CNN-GRU (Secondary) | 클래스={ckpt['num_classes']} | seq={cfg['sequence_length']}")
    return model, {'mean': mean, 'std': std}, ckpt['idx_to_label'], cfg['sequence_length']


# =============================================================================
# 3. 키포인트 추출
# =============================================================================
def _compute_finger_angles(pts_21x2):
    """손가락 관절 코사인 각도 → 15차원"""
    fingers = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
    ]
    angles = []
    for finger in fingers:
        for i in range(len(finger) - 2):
            a, b, c = pts_21x2[finger[i]], pts_21x2[finger[i+1]], pts_21x2[finger[i+2]]
            ba, bc  = a - b, c - b
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angles.append(float(np.clip(cos, -1, 1)))
    return np.array(angles, dtype=np.float32)


class DominantTracker:
    """
    매 프레임 각 손의 손목 이동량을 누적하여 dominant hand를 판별.
    EMA 방식으로 느리게 업데이트 → 세션 내내 안정적.

    반환값: 'Left' 또는 'Right' (MediaPipe 라벨 기준)
    - 초기에는 두 값이 0이므로 기본값 'Right' 반환
    - 한 손만 등장하면 자동으로 그 손이 dominant
    """
    def __init__(self, decay=0.97):
        self.decay  = decay
        self.motion = {'Left': 0.0, 'Right': 0.0}
        self.prev   = {'Left': None, 'Right': None}

    def update(self, rh):
        present = set()
        if rh.multi_hand_landmarks:
            for idx, hlm in enumerate(rh.multi_hand_landmarks):
                lab   = rh.multi_handedness[idx].classification[0].label
                wrist = np.array([hlm.landmark[0].x, hlm.landmark[0].y], dtype=np.float32)
                if self.prev[lab] is not None:
                    m = float(np.linalg.norm(wrist - self.prev[lab]))
                    # EMA 누적 (스케일 맞추기 위해 *100)
                    self.motion[lab] = (self.decay * self.motion[lab]
                                        + (1 - self.decay) * m * 100)
                self.prev[lab] = wrist
                present.add(lab)

        # 등장하지 않은 손은 서서히 감소
        for lab in ('Left', 'Right'):
            if lab not in present:
                self.motion[lab] *= self.decay

        return 'Left' if self.motion['Left'] > self.motion['Right'] else 'Right'

    @property
    def label(self):
        return 'Left' if self.motion['Left'] > self.motion['Right'] else 'Right'

    def reset(self):
        self.motion = {'Left': 0.0, 'Right': 0.0}
        self.prev   = {'Left': None, 'Right': None}


def extract_hand_relative(rh, dominant_mp_label='Right'):
    """
    MLP 입력: 114차원 [dominant(0-56) | support(57-113)]
    dominant_mp_label: DominantTracker가 판별한 MediaPipe 라벨 ('Left'/'Right')
    손 없으면 None.
    """
    dom, sup = np.zeros(57, dtype=np.float32), np.zeros(57, dtype=np.float32)
    has_hand = False

    if rh.multi_hand_landmarks:
        for idx, hlm in enumerate(rh.multi_hand_landmarks):
            lab = rh.multi_handedness[idx].classification[0].label
            pts = np.array([[lm.x, lm.y] for lm in hlm.landmark], dtype=np.float32)
            wrist = pts[0].copy()
            pts   = (pts - wrist) / (np.linalg.norm(pts[9] - wrist) + 1e-8)
            feat  = np.concatenate([pts.flatten(), _compute_finger_angles(pts)])

            if lab == dominant_mp_label:
                dom[:] = feat
            else:
                sup[:] = feat
            has_hand = True

    return np.concatenate([dom, sup]) if has_hand else None


def extract_full_keypoints(rp, rh, rf, mirror=True):
    """CNN-GRU 입력: pose75 + face210 + hand_l63 + hand_r63 = 411차원"""
    pose = np.zeros(75,  dtype=np.float32)
    face = np.zeros(210, dtype=np.float32)
    hl   = np.zeros(63,  dtype=np.float32)
    hr   = np.zeros(63,  dtype=np.float32)

    if rp and rp.pose_landmarks:
        for i, lm in enumerate(rp.pose_landmarks.landmark[:25]):
            pose[i*3], pose[i*3+1], pose[i*3+2] = lm.x, lm.y, lm.visibility
    if rf and rf.multi_face_landmarks:
        for i, lm in enumerate(rf.multi_face_landmarks[0].landmark[:70]):
            face[i*3], face[i*3+1], face[i*3+2] = lm.x, lm.y, lm.z
    if rh.multi_hand_landmarks:
        for idx, hlm in enumerate(rh.multi_hand_landmarks):
            lab = rh.multi_handedness[idx].classification[0].label
            kp  = np.array([[lm.x, lm.y, float(np.clip(lm.z * 10 + 0.5, 0, 1))]
                             for lm in hlm.landmark], dtype=np.float32).flatten()
            if mirror:
                (hr if lab == 'Left' else hl)[:] = kp
            else:
                (hl if lab == 'Left' else hr)[:] = kp

    return np.concatenate([pose, face, hl, hr])


# =============================================================================
# 4. 모션 세그멘터 (hand_rel 기반)
# =============================================================================
class MotionSeg:
    def __init__(self, thr=0.015, min_f=10, max_f=120, cool=20, smooth_w=5):
        self.thr, self.min_f, self.max_f, self.cool = thr, min_f, max_f, cool
        self.smooth_w = smooth_w
        self.active = False
        self.count  = 0   # active 프레임 수 (buf 없이 카운트만)
        self.still  = 0
        self.prev   = None
        self.motion = 0.0
        self.motion_buf = deque(maxlen=smooth_w)  # 최근 N프레임 모션 평균용

    def step(self, hand_feat):
        """hand_feat: 114-dim ndarray or None (손 없음)"""
        # 모션 계산
        self.motion = 0.0
        if self.prev is not None and hand_feat is not None:
            self.motion = float(np.linalg.norm(hand_feat - self.prev))
        if hand_feat is not None:
            self.prev = hand_feat.copy()

        self.motion_buf.append(self.motion if hand_feat is not None else 0.0)
        avg_motion = float(np.mean(self.motion_buf))  # 순간값 대신 평균으로 판단

        moving = hand_feat is not None and avg_motion > self.thr
        # 손이 사라지면 정지로 처리
        no_hand = hand_feat is None

        if not self.active:
            if moving:
                self.active, self.count, self.still = True, 1, 0
            return 'idle'

        self.count += 1
        self.still  = 0 if (moving and not no_hand) else self.still + 1

        timeout = self.count >= self.max_f
        settled = self.still >= self.cool and self.count >= self.min_f
        if timeout or settled:
            self.active, self.count, self.still = False, 0, 0
            return 'ready'
        return 'collecting'


# =============================================================================
# 5. 시퀀스 길이 조정 (CNN-GRU용)
# =============================================================================
def adjust_seq(buf, target):
    arr = np.array(buf, dtype=np.float32)
    n   = len(arr)
    if n >= target:
        s = (n - target) // 2
        return arr[s:s + target]
    pad = target - n
    pb, pa = pad // 2, pad - pad // 2
    return np.concatenate([np.tile(arr[:1], (pb, 1)), arr, np.tile(arr[-1:], (pa, 1))])


# =============================================================================
# 6. UI 헬퍼
# =============================================================================
def overlay_rect(frame, x, y, w, h, color=(15, 15, 15), alpha=0.78):
    y2, x2 = min(y + h, frame.shape[0]), min(x + w, frame.shape[1])
    roi = frame[y:y2, x:x2]
    bg  = np.full_like(roi, color[::-1])
    cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0, roi)
    frame[y:y2, x:x2] = roi


def draw_conf_bar(frame, x, y, width, height, value):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (45, 45, 45), -1)
    fw = int(width * min(max(value, 0.0), 1.0))
    if fw > 0:
        r = int(220 * (1.0 - value))
        g = int(220 * value)
        cv2.rectangle(frame, (x, y), (x + fw, y + height), (30, g, r), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)


def put_text_shadow(frame, text, pos, scale, color, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# =============================================================================
# 7. 메인 루프
# =============================================================================
def run():
    print("=" * 60)
    print("  수어 인식  |  MLP Primary + CNN-GRU Secondary")
    print("=" * 60)
    print("\n[1] 모델 로드...")

    mlp_model, mlp_norm, mlp_idx_to_label, device = load_mlp_model(MLP_MODEL_PATH)
    cnn_model, cnn_norm, cnn_idx_to_label, cnn_seq = load_cnn_model(CNN_MODEL_PATH, device)
    use_cnn = cnn_model is not None

    mp_hands = mp.solutions.hands
    mp_pose  = mp.solutions.pose
    mp_face  = mp.solutions.face_mesh
    mp_draw  = mp.solutions.drawing_utils

    hand_style = mp_draw.DrawingSpec(color=(80, 220, 120), thickness=1, circle_radius=2)
    hand_conn  = mp_draw.DrawingSpec(color=(40, 150, 80),  thickness=1)
    pose_style = mp_draw.DrawingSpec(color=(100, 160, 255), thickness=1, circle_radius=2)
    pose_conn  = mp_draw.DrawingSpec(color=(60, 100, 200),  thickness=1)

    seg      = MotionSeg(thr=MOTION_THRESHOLD, min_f=10, max_f=120, cool=20, smooth_w=5)
    dom_trk  = DominantTracker()   # dominant hand 자동 판별

    # 세그먼트 중 누적 버퍼
    mlp_seg_preds = []   # [(label, conf, probs_vec), ...]
    kp_seg_buf    = []   # full keypoints for CNN-GRU (use_cnn일 때만)

    # 라이브 MLP 윈도우 (매 프레임)
    mlp_live = deque(maxlen=MLP_VOTE_WINDOW)

    word_hist      = deque(maxlen=WORD_HISTORY_SIZE)
    last_pred_time = 0.0
    frame_idx      = 0

    display = {
        'label':      '',
        'conf':       0.0,
        'top3':       [],
        'cnn_lbl':    'N/A',
        'cnn_conf':   0.0,
        'cnn_agrees': False,
        'live_lbl':   '',
        'live_conf':  0.0,
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠 열기 실패"); return

    print("\n[2] 시작!")
    print("  q: 종료   r: 초기화\n")

    pose_ctx = mp_pose.Pose(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) if use_cnn else None
    face_ctx = mp_face.FaceMesh(max_num_faces=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) if use_cnn else None

    def process_loop(hands_ctx):
        nonlocal frame_idx, last_pred_time, dom_trk

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rh = hands_ctx.process(rgb)
            rp = pose_ctx.process(rgb) if pose_ctx else None
            rf = face_ctx.process(rgb) if face_ctx else None

            # ── Dominant hand 판별 (매 프레임 업데이트) ──────────────────
            dom_label = dom_trk.update(rh)   # 'Left' or 'Right' (MediaPipe)

            hand_rel = extract_hand_relative(rh, dominant_mp_label=dom_label)

            # ── Primary: MLP 매 프레임 추론 ────────────────────────────
            mlp_probs_now = None
            if hand_rel is not None:
                x_n = (hand_rel - mlp_norm['mean']) / mlp_norm['std']
                t   = torch.tensor(x_n, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    mlp_probs_now = torch.softmax(mlp_model(t), dim=1)[0].cpu().numpy()

                m_idx  = int(mlp_probs_now.argmax())
                m_conf = float(mlp_probs_now.max())
                m_lbl  = mlp_idx_to_label.get(m_idx, str(m_idx))

                mlp_live.append((m_lbl, m_conf))

                # 세그먼트 중이면 누적
                if seg.active:
                    mlp_seg_preds.append((m_lbl, m_conf, mlp_probs_now))

            # ── CNN-GRU용 full kp 누적 ────────────────────────────────
            if use_cnn and seg.active:
                kp_seg_buf.append(extract_full_keypoints(rp, rh, rf, mirror=True))

            # ── 모션 세그멘터 ─────────────────────────────────────────
            seg_status = seg.step(hand_rel)

            if seg_status == 'ready' and mlp_seg_preds:
                # ── MLP 세그먼트 다수결 (Primary 예측) ─────────────
                vote_counter = Counter(l for l, _, _ in mlp_seg_preds)
                best_lbl     = vote_counter.most_common(1)[0][0]
                best_confs   = [c for l, c, _ in mlp_seg_preds if l == best_lbl]
                best_conf    = float(np.mean(best_confs))

                # Top-3: 세그먼트 평균 확률 벡터
                avg_probs  = np.mean([p for _, _, p in mlp_seg_preds], axis=0)
                top3_idx   = avg_probs.argsort()[::-1][:3]
                top3       = [(mlp_idx_to_label.get(i, str(i)), float(avg_probs[i]))
                              for i in top3_idx]

                # ── CNN-GRU 확인 (Secondary) ───────────────────────
                cnn_lbl, cnn_conf, cnn_agrees = 'N/A', 0.0, False
                if use_cnn and len(kp_seg_buf) >= 5:
                    seq      = adjust_seq(kp_seg_buf, cnn_seq)
                    seq_norm = (seq - cnn_norm['mean']) / cnn_norm['std']
                    tc = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        cnn_p = torch.softmax(cnn_model(tc), dim=1)[0].cpu().numpy()
                    c_idx    = int(cnn_p.argmax())
                    cnn_conf = float(cnn_p.max())
                    cnn_lbl  = cnn_idx_to_label.get(c_idx, str(c_idx))
                    cnn_agrees = (cnn_lbl == best_lbl)

                    if cnn_agrees:
                        best_conf = min(best_conf * CNN_CONFIRM_BOOST, 1.0)
                    elif cnn_conf >= CNN_OVERRIDE_CONF and best_conf < MLP_LOW_CONF:
                        print(f"     ⚡ CNN override: {best_lbl}({best_conf*100:.0f}%) → {cnn_lbl}({cnn_conf*100:.0f}%)")
                        best_lbl  = cnn_lbl
                        best_conf = cnn_conf * 0.92

                # ── 커밋 ──────────────────────────────────────────
                now = time.time()
                if best_conf >= CONFIDENCE_THRESHOLD and \
                        now - last_pred_time >= PRED_COOLDOWN_SEC:
                    word_hist.append(best_lbl)
                    last_pred_time = now

                display.update({
                    'label':      best_lbl,
                    'conf':       best_conf,
                    'top3':       top3,
                    'cnn_lbl':    cnn_lbl,
                    'cnn_conf':   cnn_conf,
                    'cnn_agrees': cnn_agrees,
                })

                agree_mark = '✓' if cnn_agrees else ('✗' if use_cnn else '-')
                print(f"  → MLP:{best_lbl}({best_conf*100:.0f}%)"
                      f"  CNN:{cnn_lbl}({cnn_conf*100:.0f}%) [{agree_mark}]"
                      f"  [{len(mlp_seg_preds)}f]")
                print(f"     Top3: " + " | ".join(f"{l}({c*100:.0f}%)" for l, c in top3))

                mlp_seg_preds.clear()
                kp_seg_buf.clear()

            # ── 라이브 MLP 윈도우 표시용 ──────────────────────────────
            if len(mlp_live) >= 5:
                live_best = Counter(l for l, _ in mlp_live).most_common(1)[0][0]
                live_conf = float(np.mean([c for l, c in mlp_live if l == live_best]))
                display['live_lbl']  = live_best
                display['live_conf'] = live_conf

            # ── 랜드마크 그리기 ───────────────────────────────────────
            if rh.multi_hand_landmarks:
                for hl in rh.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                                           hand_style, hand_conn)
            if rp and rp.pose_landmarks:
                mp_draw.draw_landmarks(frame, rp.pose_landmarks,
                                       mp_pose.POSE_CONNECTIONS,
                                       pose_style, pose_conn)

            # ── UI ────────────────────────────────────────────────────
            H, W = frame.shape[:2]
            PAD  = 10   # 카드 내부 여백

            # ── 좌상단 메인 카드 ──────────────────────────────────────
            c   = display['conf']
            lbl = display['label'] if display['label'] else '...'

            if not display['label']:
                label_col = (110, 110, 110)
            elif c >= CONFIDENCE_THRESHOLD:
                label_col = (80, 255, 130)
            elif c >= 0.35:
                label_col = (0, 210, 255)
            else:
                label_col = (140, 130, 255)

            # 카드 크기 결정
            CARD_W, CARD_H = 230, 72
            overlay_rect(frame, 8, 8, CARD_W, CARD_H, color=(12, 12, 12), alpha=0.72)
            cv2.rectangle(frame, (8, 8), (8 + CARD_W, 8 + CARD_H), (55, 55, 55), 1)

            put_text_shadow(frame, lbl, (8 + PAD, 8 + PAD + 28), 1.1, label_col, 2)

            # 신뢰도 바 + 수치
            bar_x, bar_y = 8 + PAD, 8 + PAD + 38
            bar_w        = CARD_W - PAD * 2 - 40
            draw_conf_bar(frame, bar_x, bar_y, bar_w, 8, c)
            cv2.putText(frame, f"{c*100:.0f}%",
                        (bar_x + bar_w + 5, bar_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

            # Top-3 (카드 아래 한 줄)
            if display['top3']:
                t3 = "  ".join(f"{l} {p*100:.0f}%" for l, p in display['top3'])
                overlay_rect(frame, 8, 8 + CARD_H + 2, CARD_W, 18,
                             color=(10, 10, 10), alpha=0.65)
                cv2.putText(frame, t3, (8 + PAD, 8 + CARD_H + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

            # ── 우상단 보조 카드 (live + CNN) ─────────────────────────
            RC_W, RC_H = 180, 44
            rx = W - RC_W - 8
            overlay_rect(frame, rx, 8, RC_W, RC_H, color=(12, 12, 12), alpha=0.68)
            cv2.rectangle(frame, (rx, 8), (rx + RC_W, 8 + RC_H), (50, 50, 50), 1)

            live_lbl  = display['live_lbl']
            live_conf = display['live_conf']
            if live_lbl and hand_rel is not None:
                cv2.putText(frame, f"now: {live_lbl}  {live_conf*100:.0f}%",
                            (rx + PAD, 8 + PAD + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 185, 60), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "now: --",
                            (rx + PAD, 8 + PAD + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (70, 70, 70), 1, cv2.LINE_AA)

            if use_cnn and display['cnn_lbl'] != 'N/A':
                agree_col = (70, 210, 70) if display['cnn_agrees'] else (80, 80, 210)
                sym       = '✓' if display['cnn_agrees'] else '✗'
                cv2.putText(frame,
                            f"cnn: {display['cnn_lbl']} {display['cnn_conf']*100:.0f}% {sym}",
                            (rx + PAD, 8 + PAD + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, agree_col, 1, cv2.LINE_AA)

            # ── Dominant hand 인디케이터 (랜드마크 위에 표시) ─────────
            # dominant 손의 손목에 작은 D 마커
            if rh.multi_hand_landmarks:
                for idx_h, hlm in enumerate(rh.multi_hand_landmarks):
                    lab = rh.multi_handedness[idx_h].classification[0].label
                    if lab == dom_label:
                        wx = int(hlm.landmark[0].x * W)
                        wy = int(hlm.landmark[0].y * H)
                        cv2.circle(frame, (wx, wy), 10, (0, 220, 255), 2)
                        cv2.putText(frame, 'D', (wx - 5, wy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

            # ── 하단 바 ───────────────────────────────────────────────
            BOT_H = 28
            overlay_rect(frame, 0, H - BOT_H, W, BOT_H, color=(8, 8, 8), alpha=0.80)

            # 녹화 인디케이터
            if seg.active:
                blink   = (frame_idx // 12) % 2 == 0
                dot_col = (0, 255, 60) if blink else (0, 110, 30)
                cv2.circle(frame, (16, H - 13), 5, dot_col, -1)
                cv2.putText(frame, f"REC {seg.count}f",
                            (26, H - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 60), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (16, H - 13), 5, (50, 50, 50), -1)
                cv2.putText(frame, "READY",
                            (26, H - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)

            # 모션 강도
            mv  = min(seg.motion / (MOTION_THRESHOLD * 4), 1.0)
            mbx = 90
            cv2.rectangle(frame, (mbx, H - 21), (mbx + 60, H - 13), (35, 35, 35), -1)
            if int(60 * mv) > 0:
                cv2.rectangle(frame, (mbx, H - 21),
                              (mbx + int(60 * mv), H - 13), (0, 165, 255), -1)

            # 단어 히스토리 (바 중앙)
            if word_hist:
                hist_str = "  ›  ".join(word_hist)
                cv2.putText(frame, hist_str,
                            (W // 2 - 100, H - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (155, 205, 155), 1, cv2.LINE_AA)

            # 단축키 (우하단)
            cv2.putText(frame, "r  q",
                        (W - 38, H - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (75, 75, 75), 1, cv2.LINE_AA)

            cv2.imshow("수어 인식", frame)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                seg.active = False; seg.count = 0; seg.still = 0
                mlp_seg_preds.clear(); kp_seg_buf.clear()
                mlp_live.clear(); word_hist.clear()
                dom_trk.reset()
                last_pred_time = 0.0
                display.update({'label': '', 'conf': 0.0, 'top3': [],
                                 'cnn_lbl': 'N/A', 'cnn_conf': 0.0,
                                 'cnn_agrees': False, 'live_lbl': '', 'live_conf': 0.0})
                print("  초기화 완료 (dominant 추적 리셋)")

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands_ctx:
        if pose_ctx:
            with pose_ctx, face_ctx:
                process_loop(hands_ctx)
        else:
            process_loop(hands_ctx)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
