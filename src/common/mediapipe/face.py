import cv2
import mediapipe as mp

# MediaPipe 솔루션 모듈
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_connections = mp.solutions.face_mesh_connections

# BGR 자주색
PURPLE = (255, 0, 255)


# 콧대 168 -> 4 ( 총 6개 )
NOSE_CONNECTIONS = [
    (168, 6),
    (6, 197),
    (197, 195),
    (195, 5),
    (5, 4),
]

# 인중 부분 98 -> 327 ( 총 5개 )
PHILTRUM_CONNECTIONS = [
    (98, 97),
    (97, 2),
    (2, 326),
    (326, 327),
]

#눈썹 => 위에 부분만 (총 10개 )
RIGHT_EYEBROW_POINTS = [70, 63, 105, 66, 107]
LEFT_EYEBROW_POINTS = [336, 296, 334, 293, 300]

#얼굴 => 귀 부분 까지만 152 - 162~389 (총 25개) 
FACE_OUTLINE_POINTS = [
    162,
    127,
    234,
    93,
    132,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
    152,
    377,
    400,
    378,
    379,
    365,
    397,
    288,
    361,
    323,
    454,
    356,
    389,
]


# ---- Drawing Helpers ----------------------------------------------------- #
def draw_lines(image, landmarks, img_h, img_w, connections, thickness):
    
    # 픽셀 좌표로 변환 + 색 => 선 그림 
    # 이미지 사이즈를 곱해 실제 좌표((0,0)~(1,1))로 변환.
    
    for start, end in connections:
        if start >= len(landmarks) or end >= len(landmarks):
            continue

        start = landmarks[start]
        end = landmarks[end]

        p1 = int(start.x * img_w), int(start.y * img_h)
        p2 = int(end.x * img_w), int(end.y * img_h)

        cv2.line(image, p1, p2, PURPLE, thickness)


def connections_list(points): # 포인트 -> 선분 리스트 

    connections = []
    for idx in range(len(points) - 1):

        start = points[idx]
        end = points[idx + 1]
        connections.append((start, end))

    return connections


FACE_OUTLINE_CONNECTIONS = connections_list(FACE_OUTLINE_POINTS)
RIGHT_EYEBROW_CONNECTIONS = connections_list(RIGHT_EYEBROW_POINTS)
LEFT_EYEBROW_CONNECTIONS = connections_list(LEFT_EYEBROW_POINTS)



def draw_face(image, results, line_thickness=2):
   
    if not results or not results.multi_face_landmarks:
        return image

    img_h, img_w = image.shape[:2]
    connection_groups = [
        mp_connections.FACEMESH_RIGHT_EYE,
        mp_connections.FACEMESH_LEFT_EYE,
        mp_connections.FACEMESH_LIPS,
    ]
    connection_style = mp_drawing.DrawingSpec(
        color=PURPLE, thickness=line_thickness, circle_radius=0
    )

    for face_landmarks in results.multi_face_landmarks: #n명의 얼굴이 잡혔을때 처리
        for connections in connection_groups: # 눈,코,인중,입술,오른쪽 눈썹, 왼쪽 눈썹을 순회.
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=connection_style,
            )

        draw_lines(
            image,
            face_landmarks.landmark,
            img_h,
            img_w,
            NOSE_CONNECTIONS,
            line_thickness,
        )
        draw_lines(
            image,
            face_landmarks.landmark,
            img_h,
            img_w,
            PHILTRUM_CONNECTIONS,
            line_thickness,
        )
        draw_lines(
            image,
            face_landmarks.landmark,
            img_h,
            img_w,
            RIGHT_EYEBROW_CONNECTIONS,
            line_thickness,
        )
        draw_lines(
            image,
            face_landmarks.landmark,
            img_h,
            img_w,
            LEFT_EYEBROW_CONNECTIONS,
            line_thickness,
        )
        if FACE_OUTLINE_CONNECTIONS:
            draw_lines(
                image,
                face_landmarks.landmark,
                img_h,
                img_w,
                FACE_OUTLINE_CONNECTIONS,
                line_thickness,
            )

    return image


# ---- Webcam Demo --------------------------------------------------------- #
def video_origin():
    """
    기본 웹캠(0번)을 열어 좌우 반전된 프레임에 FaceMesh를 적용하고 draw_face 결과를 실시간으로 표시한다.
    'q' 키 입력 시 캡처 루프를 종료하고 자원을 해제한다.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True, # F 기본적인 지점 468개 | T 홍채+ 입주변 총 478개 제공 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        # 위 설정값들은 실시간 추적에 적당한 기본 민감도로 조정한다.
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            out_frame = draw_face(frame, results, line_thickness=2)
            cv2.imshow("Face contours (purple)", out_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_origin()
