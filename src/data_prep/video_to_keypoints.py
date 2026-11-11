"""
직접영상제작 폴더의 동영상을 MediaPipe로 처리하여 키포인트 JSON 파일로 변환

============================================================
실행 방법 (터미널):
============================================================
1. 가상환경 활성화:
   conda activate py311_env

2. 스크립트 실행 (레포 루트에서):
   python src/data_prep/video_to_keypoints.py

============================================================
실행 전 설정 (맨 아래 if __name__ == "__main__": 부분):
============================================================
- video_folder: 입력 동영상 폴더 경로
- output_base_folder: 출력 JSON 폴더 경로
- skip_mode: 빈 프레임 필터 모드 ("all", "hands", "hands_or_face", "none")
============================================================
"""
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import glob
import re
from pathlib import Path

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
OUTPUT_DIR = str(_ROOT / "dataset/자체제작_단어_keypoints")


#conda activate py311_env
#


# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


def extract_label_from_filename(filename):
    """
    파일명에서 순수 라벨(단어)만 추출
    
    예시:
        많다_1(2).mp4 → 많다
        안녕하세요_1(2).mp4 → 안녕하세요
        가다1.mp4 → 가다
        
    Args:
        filename: 파일명 (확장자 포함)
    
    Returns:
        str: 추출된 라벨명
    """
    # 확장자 제거
    name_without_ext = os.path.splitext(filename)[0]
    # 숫자, 언더스코어, 괄호 기준으로 분리하여 첫 번째 부분만 사용
    label = re.split(r'[_\d(]', name_without_ext)[0]
    return label


def is_empty_keypoints(keypoint_data, mode="all"):
    """
    키포인트 데이터가 비어있는지 확인

    Args:
        keypoint_data: 키포인트 딕셔너리
        mode: 빈 프레임 판단 기준
            - "all": 모든 키포인트가 0일 때만 빈 프레임 (기본값)
            - "hands": 양손 모두 감지 안 되면 빈 프레임
            - "hands_or_face": 손과 얼굴 모두 감지 안 되면 빈 프레임
            - "none": 필터링 안 함 (모든 프레임 저장)

    Returns:
        bool: 빈 프레임이면 True
    """
    if mode == "none":
        return False

    if not keypoint_data["people"]:
        return True

    person = keypoint_data["people"][0]

    if mode == "all":
        # 모든 키포인트가 0인지 확인
        all_keypoints = (
            person["pose_keypoints_2d"] +
            person["face_keypoints_2d"] +
            person["hand_left_keypoints_2d"] +
            person["hand_right_keypoints_2d"]
        )
        return all(v == 0.0 for v in all_keypoints)

    elif mode == "hands":
        # 손 키포인트만 확인 (왼손 또는 오른손 감지되면 유효)
        left_detected = any(v != 0.0 for v in person["hand_left_keypoints_2d"])
        right_detected = any(v != 0.0 for v in person["hand_right_keypoints_2d"])
        return not (left_detected or right_detected)

    elif mode == "hands_or_face":
        # 손 또는 얼굴이 감지되면 유효
        left_detected = any(v != 0.0 for v in person["hand_left_keypoints_2d"])
        right_detected = any(v != 0.0 for v in person["hand_right_keypoints_2d"])
        face_detected = any(v != 0.0 for v in person["face_keypoints_2d"])
        return not (left_detected or right_detected or face_detected)

    return False


def extract_keypoints_from_frame(frame, pose, hands, face_mesh):
    """
    단일 프레임에서 키포인트 추출 (model.ipynb의 411차원 형식)

    Returns:
        dict: OpenPose 형식의 키포인트 데이터
              {
                  "people": [{
                      "pose_keypoints_2d": [75개],    # 25 landmarks * 3 (x, y, confidence)
                      "face_keypoints_2d": [210개],   # 70 landmarks * 3
                      "hand_left_keypoints_2d": [63개],   # 21 landmarks * 3
                      "hand_right_keypoints_2d": [63개]   # 21 landmarks * 3
                  }]
              }
    """
    # BGR → RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe 처리
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    # 1. Pose keypoints (25 landmarks * 3 = 75)
    pose_kps = []
    if pose_results.pose_landmarks:
        for i in range(25):  # 처음 25개 랜드마크만 사용
            if i < len(pose_results.pose_landmarks.landmark):
                lm = pose_results.pose_landmarks.landmark[i]
                pose_kps.extend([lm.x, lm.y, lm.visibility])
            else:
                pose_kps.extend([0.0, 0.0, 0.0])
    else:
        pose_kps = [0.0] * 75

    # 2. Face keypoints (70 landmarks * 3 = 210)
    face_kps = []
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        for i in range(70):  # 처음 70개 랜드마크만 사용
            if i < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[i]
                face_kps.extend([lm.x, lm.y, lm.z])
            else:
                face_kps.extend([0.0, 0.0, 0.0])
    else:
        face_kps = [0.0] * 210

    # 3. Hand keypoints (21 * 3 * 2 = 126)
    left_hand_kps = [0.0] * 63
    right_hand_kps = [0.0] * 63

    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            handedness = hands_results.multi_handedness[idx].classification[0].label
            hand_kps = []
            for lm in hand_landmarks.landmark:
                hand_kps.extend([lm.x, lm.y, lm.z])

            if handedness == "Left":
                left_hand_kps = hand_kps
            elif handedness == "Right":
                right_hand_kps = hand_kps

    # OpenPose 형식으로 반환 (총 75 + 210 + 63 + 63 = 411)
    keypoint_data = {
        "people": [{
            "pose_keypoints_2d": pose_kps,
            "face_keypoints_2d": face_kps,
            "hand_left_keypoints_2d": left_hand_kps,
            "hand_right_keypoints_2d": right_hand_kps
        }]
    }

    return keypoint_data


def process_video_to_json(video_path, output_folder, label_name, video_id, skip_mode="all"):
    """
    동영상을 프레임별로 처리하여 JSON 키포인트 파일들로 저장

    Args:
        video_path: 입력 동영상 경로
        output_folder: 출력 폴더 경로
        label_name: 라벨 이름 (예: '많다', '안녕하세요', ...)
        video_id: 동영상 고유 ID (같은 라벨의 여러 영상 구분용)
        skip_mode: 빈 프레임 건너뛰기 기준
            - "all": 모든 키포인트가 0일 때만 건너뛰기 (기본값)
            - "hands": 손이 감지 안 되면 건너뛰기
            - "hands_or_face": 손과 얼굴 모두 없으면 건너뛰기
            - "none": 건너뛰지 않음 (모든 프레임 저장)
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 동영상 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"동영상을 열 수 없습니다: {video_path}")
        return False

    # 동영상 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n 처리 중: {label_name} (영상 ID: {video_id})")
    print(f"   경로: {video_path}")
    print(f"   FPS: {fps}, 총 프레임: {total_frames}")

    # MediaPipe 초기화
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        frame_idx = 0
        saved_frames = 0
        skipped_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 키포인트 추출
            keypoint_data = extract_keypoints_from_frame(frame, pose, hands, face_mesh)

            # 빈 프레임 체크 및 건너뛰기
            if is_empty_keypoints(keypoint_data, mode=skip_mode):
                frame_idx += 1
                skipped_frames += 1
                continue

            # JSON 파일로 저장 (라벨_영상ID_프레임번호 형식)
            json_filename = f"{label_name}_{video_id}_{frame_idx:06d}_keypoints.json"
            json_path = os.path.join(output_folder, json_filename)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(keypoint_data, f, ensure_ascii=False, indent=2)

            frame_idx += 1
            saved_frames += 1

            # 진행 상황 표시
            if frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"   진행: {frame_idx}/{total_frames} ({progress:.1f}%)", end='\r')

        cap.release()
        print(f"\n   완료: {saved_frames}개 저장 / {skipped_frames}개 건너뜀 (총 {frame_idx}프레임)")
        return True


def create_morpheme_label_file(output_folder, label_name):
    """
    model.ipynb에서 사용하는 형식의 라벨 JSON 파일 생성

    형식:
    {
        "data": [{
            "attributes": [{
                "name": "많다"
            }]
        }]
    }
    """
    label_data = {
        "data": [{
            "attributes": [{
                "name": label_name
            }]
        }]
    }

    # 라벨 파일 저장
    label_filename = f"{label_name}_morpheme.json"
    label_path = os.path.join(output_folder, label_filename)

    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, ensure_ascii=False, indent=2)

    print(f"라벨 파일 생성: {label_filename}")


def process_all_videos(video_folder, output_base_folder, skip_mode="all"):
    """
    직접영상제작 폴더의 모든 동영상을 처리

    Args:
        video_folder: 입력 동영상 폴더 (직접영상제작)
        output_base_folder: 출력 기본 폴더
        skip_mode: 빈 프레임 건너뛰기 기준
            - "all": 모든 키포인트가 0일 때만 건너뛰기 (기본값)
            - "hands": 손이 감지 안 되면 건너뛰기
            - "hands_or_face": 손과 얼굴 모두 없으면 건너뛰기
            - "none": 건너뛰지 않음 (모든 프레임 저장)
    """
    # 동영상 파일 찾기
    video_files = glob.glob(os.path.join(video_folder, "**", "*.mov"), recursive=True)
    video_files.extend(glob.glob(os.path.join(video_folder, "**", "*.mp4"), recursive=True))
    video_files = sorted(video_files)

    if not video_files:
        print(" 동영상 파일을 찾을 수 없습니다!")
        return

    skip_mode_desc = {
        "all": "모든 키포인트가 0일 때만 건너뛰기",
        "hands": "손이 감지 안 되면 건너뛰기",
        "hands_or_face": "손과 얼굴 모두 없으면 건너뛰기",
        "none": "건너뛰지 않음 (모든 프레임 저장)"
    }

    print(f"\n{'='*70}")
    print(f"직접영상제작 폴더 동영상 → JSON 키포인트 변환")
    print(f"{'='*70}")
    print(f"입력 폴더: {video_folder}")
    print(f"출력 폴더: {output_base_folder}")
    print(f"총 동영상: {len(video_files)}개")
    print(f"빈 프레임 필터: {skip_mode} ({skip_mode_desc.get(skip_mode, '알 수 없음')})")
    print(f"{'='*70}\n")

    # 라벨별 영상 카운터 (같은 라벨의 여러 영상 구분용)
    label_video_counter = {}
    
    # 각 동영상 처리
    success_count = 0
    fail_count = 0

    for idx, video_path in enumerate(video_files, 1):
        # 파일명에서 라벨 추출
        filename = os.path.basename(video_path)
        label_name = extract_label_from_filename(filename)
        
        # 같은 라벨의 영상 번호 부여
        if label_name not in label_video_counter:
            label_video_counter[label_name] = 0
        label_video_counter[label_name] += 1
        video_id = label_video_counter[label_name]

        # 출력 폴더: output_base_folder/label_name/
        output_folder = os.path.join(output_base_folder, label_name)

        print(f"\n[{idx}/{len(video_files)}] {label_name} (원본: {filename})")

        # 동영상 처리
        success = process_video_to_json(video_path, output_folder, label_name, video_id, skip_mode)

        if success:
            success_count += 1
        else:
            fail_count += 1

    # 각 라벨 폴더에 morpheme.json 생성 (한 번만)
    print(f"\n라벨 파일 생성 중...")
    for label_name in label_video_counter.keys():
        output_folder = os.path.join(output_base_folder, label_name)
        create_morpheme_label_file(output_folder, label_name)

    # 결과 요약
    print(f"\n{'='*70}")
    print(f"변환 완료!")
    print(f"{'='*70}")
    print(f" 성공: {success_count}개")
    print(f" 실패: {fail_count}개")
    print(f" 총 라벨 수: {len(label_video_counter)}개")
    print(f"\n라벨별 영상 수:")
    for label, count in sorted(label_video_counter.items()):
        print(f"   {label}: {count}개 영상")
    print(f"{'='*70}")


if __name__ == "__main__":
    # ============================================================
    # 처리할 폴더 목록
    # ============================================================
    # 원본 영상 폴더: raw/videos/ 아래의 하위 폴더를 전부 처리합니다.
    RAW_VIDEO_DIR = _ROOT / "raw" / "videos"
    if not RAW_VIDEO_DIR.exists():
        print(f"[ERROR] 원본 영상 폴더가 없습니다: {RAW_VIDEO_DIR}")
        exit(1)
    jobs = [
        {"video_folder": str(folder), "output_base_folder": OUTPUT_DIR}
        for folder in sorted(RAW_VIDEO_DIR.iterdir()) if folder.is_dir()
    ]

    # ============================================================
    # 빈 프레임 필터 모드 설정
    # ============================================================
    # "all"           : 모든 키포인트가 0일 때만 건너뛰기 (기본값)
    # "hands"         : 손이 감지 안 되면 건너뛰기
    # "hands_or_face" : 손과 얼굴 모두 없으면 건너뛰기
    # "none"          : 건너뛰지 않음 (모든 프레임 저장)
    # ============================================================
    skip_mode = "hands"

    for job in jobs:
        video_folder = job["video_folder"]
        output_base_folder = job["output_base_folder"]

        print(f"\n{'='*70}")
        print(f"처리 시작: {video_folder}")
        print(f"{'='*70}")

        if not os.path.exists(video_folder):
            print(f"⚠️  입력 폴더가 존재하지 않습니다: {video_folder}")
            continue

        process_all_videos(video_folder, output_base_folder, skip_mode)

    print(f"\n\n모든 폴더 처리 완료!")
    print(f"\n다음 단계:")
    print(f"1. 생성된 JSON 파일 확인: {OUTPUT_DIR}")
    print(f"2. model.ipynb에서 label_base_dir, keypoint_base_dir 경로 수정")
    print(f"3. model.ipynb 실행하여 학습 데이터 생성")