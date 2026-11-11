# -*- coding: utf-8 -*-
"""
직접사진제작_지문자 폴더의 사진을 MediaPipe로 처리하여 키포인트 JSON 파일로 변환
model.ipynb의 데이터 로딩 방식과 동일한 구조로 저장
"""
import cv2
import mediapipe as mp
import json
import os
import glob

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


# conda activate py311_env


# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


def extract_keypoints_from_image(image, pose, hands, face_mesh):
    """
    단일 이미지에서 키포인트 추출 (model.ipynb의 411차원 형식)
    """
    # BGR -> RGB 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # MediaPipe 처리
    pose_results = pose.process(rgb_image)
    hands_results = hands.process(rgb_image)
    face_results = face_mesh.process(rgb_image)

    # 1. Pose keypoints (25 landmarks * 3 = 75)
    pose_kps = []
    if pose_results.pose_landmarks:
        for i in range(25):
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
        for i in range(70):
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


def process_image_to_json(image_path, output_folder, label_name):
    """
    이미지를 처리하여 JSON 키포인트 파일로 저장
    """
    os.makedirs(output_folder, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot open image: {image_path}")
        return False

    print(f"\n[PHOTO] Processing: {label_name}")
    print(f"   Path: {image_path}")
    print(f"   Size: {image.shape[1]}x{image.shape[0]}")

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose, mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as hands, mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        keypoint_data = extract_keypoints_from_image(image, pose, hands, face_mesh)

        json_filename = f"{label_name}_000000_keypoints.json"
        json_path = os.path.join(output_folder, json_filename)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(keypoint_data, f, ensure_ascii=False, indent=2)

        print(f"   [OK] Done: {json_filename}")
        return True


def create_morpheme_label_file(output_folder, label_name):
    """
    model.ipynb에서 사용하는 형식의 라벨 JSON 파일 생성
    """
    label_data = {
        "data": [{
            "attributes": [{
                "name": label_name
            }]
        }]
    }

    label_filename = f"{label_name}_morpheme.json"
    label_path = os.path.join(output_folder, label_filename)

    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, ensure_ascii=False, indent=2)

    print(f"   [LABEL] Created: {label_filename}")


def process_folder(subfolder_path, output_base_folder, folder_name):
    """
    특정 하위 폴더(1, 2, 3)의 이미지를 처리
    """
    image_files = glob.glob(os.path.join(subfolder_path, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(subfolder_path, "*.jpeg")))
    image_files.extend(glob.glob(os.path.join(subfolder_path, "*.png")))
    image_files = sorted(image_files)

    if not image_files:
        print(f"[ERROR] No image files found in {subfolder_path}!")
        return 0, 0

    print(f"\n{'='*70}")
    print(f"Processing folder: {folder_name}")
    print(f"{'='*70}")
    print(f"Input folder: {subfolder_path}")
    print(f"Output folder: {output_base_folder}")
    print(f"Total images: {len(image_files)}")
    print(f"{'='*70}\n")

    success_count = 0
    fail_count = 0

    for idx, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        label_name = filename.split('_')[0]

        output_folder = os.path.join(output_base_folder, label_name)

        print(f"\n[{idx}/{len(image_files)}] {label_name}")

        success = process_image_to_json(image_path, output_folder, label_name)

        if success:
            create_morpheme_label_file(output_folder, label_name)
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def process_all_images(image_folder, output_base_folder):
    """
    직접사진제작_지문자 폴더의 1, 2, 3 하위폴더를 각각 처리
    """
    subfolders = ['1', '2', '3']
    total_success = 0
    total_fail = 0

    for folder_name in subfolders:
        subfolder_path = os.path.join(image_folder, folder_name)
        if not os.path.exists(subfolder_path):
            print(f"[WARNING] Folder not found: {subfolder_path}")
            continue

        # 각 폴더별로 별도의 출력 폴더 생성
        folder_output = os.path.join(output_base_folder, folder_name)

        success, fail = process_folder(subfolder_path, folder_output, folder_name)
        total_success += success
        total_fail += fail

    print(f"\n{'='*70}")
    print(f"All Conversions Complete!")
    print(f"{'='*70}")
    print(f"[OK] Total Success: {total_success}")
    print(f"[FAIL] Total Failed: {total_fail}")
    print(f"\nOutput structure:")
    print(f"  {output_base_folder}/")
    print(f"  ├── 1/")
    print(f"  │   ├── ㄱ/")
    print(f"  │   │   ├── ㄱ_000000_keypoints.json")
    print(f"  │   │   └── ㄱ_morpheme.json")
    print(f"  │   └── ...")
    print(f"  ├── 2/")
    print(f"  │   └── ...")
    print(f"  └── 3/")
    print(f"      └── ...")
    print(f"{'='*70}")


if __name__ == "__main__":
    image_folder = str(_ROOT / "raw/photos")
    output_base_folder = str(_ROOT / "dataset/자체제작_사진_지문자_keypoints")

    if not os.path.exists(image_folder):
        print(f"[ERROR] Input folder does not exist: {image_folder}")
        exit(1)

    process_all_images(image_folder, output_base_folder)

    print(f"\n\nNext steps:")
    print(f"1. Check generated JSON files: {output_base_folder}")
    print(f"2. Update label_base_dir, keypoint_base_dir in model.ipynb")
    print(f"3. Run model.ipynb to generate training data")