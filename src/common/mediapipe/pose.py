import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# Pose 연결 중에서 0~24번 랜드마크끼리만 사용 -> 데이터 조건이 25의 포인트만 사용해서 이렇게 잡음 
# 나오지 않은 나머지는 다리 쪽이라 쓸일 없는 부분임. 
POSE_CONNECTIONS_HEAD_TO_KNEE = [
    (start, end)
    for start, end in mp_pose.POSE_CONNECTIONS
    if start <= 24 and end <= 24
]


def draw_pose(image, results, point_radius=4, line_thickness=2):
    
    if not results or not results.pose_landmarks:
        return image

    h, w = image.shape[:2] #0 : 너비 , 1: 높이 
    landmarks = results.pose_landmarks.landmark

    # 선(검정색)
    for start, end in POSE_CONNECTIONS_HEAD_TO_KNEE:
        start_lm = landmarks[start]
        end_lm = landmarks[end]
        x1 = int(start_lm.x * w)
        y1 = int(start_lm.y * h)
        x2 = int(end_lm.x * w)
        y2 = int(end_lm.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), line_thickness)

    # 포인트(빨강)
    for idx in range(25):
        lm = landmarks[idx]
        px = int(lm.x * w)
        py = int(lm.y * h)
        cv2.circle(image, (px, py), point_radius, (0, 0, 255), -1)
        cv2.putText(
            image,
            str(idx),
            (px + 5, py -5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return image


def Vedio_origin():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    with mp_pose.Pose(
        static_image_mode=False, # F가 영상 , T가 사진 
        model_complexity=1, # 아직 2가 지원되지 않음. 해봤는데 에러남. 
        enable_segmentation=False, # F 사람 /  T 사람 + 사람 실루엣 
        min_detection_confidence=0.5,# 손바닥 탐지 
        min_tracking_confidence=0.5,#손 추적 
    ) as pose:
        


        while True:
            ret, frame = cap.read()
            if not ret:
                break

            #셀카처럼 보이게 좌우 반전 
            frame = cv2.flip(frame, 1)

            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            out = draw_pose(frame, results, point_radius=4, line_thickness=2)
            cv2.imshow("Pose (points: red, lines: black)", out)

            if cv2.waitKey(1) & 0xFF == ord("q"): #종료조건 2
                break

    cap.release()
    cv2.destroyAllWindows() #gui 종료 


if __name__ == "__main__":
    Vedio_origin()
