# =============================================================================
# diagnose.py
# 학습 데이터(JSON)와 실시간 MediaPipe 출력이 얼마나 다른지 진단
# 실행: python diagnose.py
# =============================================================================

import cv2
import numpy as np
import json
import glob
import os
import torch
import torch.nn as nn
import mediapipe as mp

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

# ── 설정 ──────────────────────────────────────────────────
DATA_DIR      = str(_ROOT / "dataset/자체제작_단어_keypoints")
AUX_MODEL     = str(_ROOT / "models/video_model/frame_mlp.pt")
MAIN_MODEL    = str(_ROOT / "models/video_model/sign_language_cnn_gru.pt")

# 진단할 클래스 3개 (실제 존재하는 이름으로 수정하세요)
TEST_LABELS   = ["가다", "감사합니다", "사랑"]
# ──────────────────────────────────────────────────────────


# ── FrameMLP 직접 정의 (외부 파일 불필요) ─────────────────
class FrameMLP(nn.Module):
    def __init__(self, input_dim=84, hidden_dim=256, num_classes=271):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.net(x)
# ──────────────────────────────────────────────────────────


def normalize_hand_rel(pts_21x2):
    """손목 기준 상대좌표 + 스케일 정규화 → 42차원"""
    wrist = pts_21x2[0].copy()
    pts   = pts_21x2 - wrist
    scale = np.linalg.norm(pts[9]) + 1e-8
    return (pts / scale).flatten()


def load_json_hand_sample(data_dir, label):
    """JSON 폴더에서 해당 라벨의 첫 번째 유효 프레임 로드"""
    folders = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and label in d
    ])
    if not folders:
        return None, None

    for folder in folders[:3]:
        jsons = sorted(glob.glob(os.path.join(data_dir, folder, '*.json')))
        for jp in jsons:
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'people' not in data or not data['people']:
                    continue
                p = data['people'][0]
                raw_l = p.get('hand_left_keypoints_2d', [])
                raw_r = p.get('hand_right_keypoints_2d', [])

                # 절대 좌표
                hl_abs = np.array(raw_l[:63], dtype=np.float32) if raw_l else np.zeros(63)
                hr_abs = np.array(raw_r[:63], dtype=np.float32) if raw_r else np.zeros(63)

                # 상대 좌표
                hl_rel = np.zeros(42, dtype=np.float32)
                hr_rel = np.zeros(42, dtype=np.float32)

                if raw_l and len(raw_l) >= 2 and max(raw_l[:2]) > 0.01:
                    pts_l = np.array(raw_l).reshape(-1, 3)[:21, :2]
                    hl_rel = normalize_hand_rel(pts_l)

                if raw_r and len(raw_r) >= 2 and max(raw_r[:2]) > 0.01:
                    pts_r = np.array(raw_r).reshape(-1, 3)[:21, :2]
                    hr_rel = normalize_hand_rel(pts_r)

                if hl_rel.max() == 0 and hr_rel.max() == 0:
                    continue

                return (np.concatenate([hl_abs, hr_abs]),
                        np.concatenate([hl_rel, hr_rel]))
            except Exception:
                continue
    return None, None


def capture_realtime_sample(label_name):
    """웹캠에서 실시간으로 손 키포인트 캡처 (스페이스바로 캡처)"""
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    result_abs = None
    result_rel = None

    print(f"\n  [{label_name}] 수어를 하고 스페이스바를 누르세요. q=건너뜀")

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)

            hl_abs = np.zeros(63, dtype=np.float32)
            hr_abs = np.zeros(63, dtype=np.float32)
            hl_rel = np.zeros(42, dtype=np.float32)
            hr_rel = np.zeros(42, dtype=np.float32)
            hand_found = False

            if res.multi_hand_landmarks:
                for idx, hlm in enumerate(res.multi_hand_landmarks):
                    lab = res.multi_handedness[idx].classification[0].label
                    mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

                    kp_abs = np.array([[lm.x, lm.y, float(np.clip(lm.z*10+0.5,0,1))]
                                       for lm in hlm.landmark], dtype=np.float32).flatten()
                    pts_2d = np.array([[lm.x, lm.y] for lm in hlm.landmark], dtype=np.float32)
                    kp_rel = normalize_hand_rel(pts_2d)

                    # Mirror 모드 기준
                    if lab == 'Left':
                        hr_abs[:] = kp_abs; hr_rel[:] = kp_rel
                    else:
                        hl_abs[:] = kp_abs; hl_rel[:] = kp_rel
                    hand_found = True

            status = "손 감지됨 ✓" if hand_found else "손 없음 ✗"
            col    = (0, 255, 0) if hand_found else (0, 0, 255)
            cv2.putText(frame, f"[{label_name}] {status}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.putText(frame, "SPACE: 캡처  Q: 건너뜀",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.imshow("Diagnose - Realtime", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and hand_found:
                result_abs = np.concatenate([hl_abs, hr_abs])
                result_rel = np.concatenate([hl_rel, hr_rel])
                print(f"  ✓ 캡처 완료")
                break
            elif key == ord('q'):
                print(f"  건너뜀")
                break

    cap.release()
    cv2.destroyAllWindows()
    return result_abs, result_rel


def run_diagnosis():
    print("=" * 60)
    print("  진단 스크립트: 학습 데이터 vs 실시간 입력 비교")
    print("=" * 60)

    # 보조 모델 로드
    ckpt     = torch.load(AUX_MODEL, map_location='cpu', weights_only=False)
    aux_mean = np.array(ckpt['norm_params']['mean']).flatten()
    aux_std  = np.array(ckpt['norm_params']['std']).flatten()
    aux_std  = np.where(aux_std < 1e-8, 1.0, aux_std)
    idx_to_label = ckpt['idx_to_label']
    input_dim    = ckpt.get('input_dim', 84)

    aux_model = FrameMLP(input_dim=input_dim,
                         hidden_dim=ckpt.get('hidden_dim', 256),
                         num_classes=ckpt['num_classes'])
    aux_model.load_state_dict(ckpt['model_state_dict'])
    aux_model.eval()

    print(f"\n보조 모델 입력 차원: {input_dim}")
    print(f"정규화 mean 범위: [{aux_mean.min():.3f}, {aux_mean.max():.3f}]")
    print(f"정규화 std  범위: [{aux_std.min():.3f}, {aux_std.max():.3f}]")

    print("\n" + "─" * 60)
    print("  각 수어에 대해 JSON 샘플과 실시간 샘플을 비교합니다.")
    print("─" * 60)

    for label in TEST_LABELS:
        print(f"\n▶ [{label}]")

        # JSON 샘플
        json_abs, json_rel = load_json_hand_sample(DATA_DIR, label)
        if json_rel is None:
            print(f"  JSON 샘플 없음 → TEST_LABELS 이름을 데이터 폴더명과 맞춰주세요")
            continue

        print(f"  JSON 절대좌표 범위: [{json_abs.min():.3f}, {json_abs.max():.3f}]")
        print(f"  JSON 상대좌표 범위: [{json_rel.min():.3f}, {json_rel.max():.3f}]")

        # JSON으로 보조 모델 예측
        json_in = (json_rel[:input_dim] - aux_mean) / aux_std
        with torch.no_grad():
            p = torch.softmax(aux_model(torch.tensor(json_in, dtype=torch.float32).unsqueeze(0)), dim=1)[0]
        top3 = p.topk(3)
        print(f"  JSON 입력 → 예측 Top3:")
        for v, i in zip(top3.values, top3.indices):
            marker = " ← 정답!" if idx_to_label[i.item()] == label else ""
            print(f"    {idx_to_label[i.item()]:15s} {v.item()*100:.1f}%{marker}")

        # 실시간 샘플
        rt_abs, rt_rel = capture_realtime_sample(label)
        if rt_rel is None:
            continue

        print(f"  실시간 절대좌표 범위: [{rt_abs.min():.3f}, {rt_abs.max():.3f}]")
        print(f"  실시간 상대좌표 범위: [{rt_rel.min():.3f}, {rt_rel.max():.3f}]")

        # 코사인 유사도
        a, b = json_rel[:input_dim], rt_rel[:input_dim]
        sim  = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        print(f"  JSON vs 실시간 코사인 유사도: {sim:.3f}  {'✓ 유사함' if sim > 0.7 else '✗ 많이 다름 ← 문제!'}")

        # 실시간으로 보조 모델 예측
        rt_in = (rt_rel[:input_dim] - aux_mean) / aux_std
        with torch.no_grad():
            p2 = torch.softmax(aux_model(torch.tensor(rt_in, dtype=torch.float32).unsqueeze(0)), dim=1)[0]
        top3_rt = p2.topk(3)
        print(f"  실시간 입력 → 예측 Top3:")
        for v, i in zip(top3_rt.values, top3_rt.indices):
            marker = " ← 정답!" if idx_to_label[i.item()] == label else ""
            print(f"    {idx_to_label[i.item()]:15s} {v.item()*100:.1f}%{marker}")

    print("\n" + "=" * 60)
    print("  진단 완료")
    print("  코사인 유사도가 0.7 미만 → 손 방향(Mirror/Normal) 문제")
    print("  JSON 예측도 틀림 → 모델 자체 문제, 재학습 필요")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnosis()