import cv2
import mediapipe as mp

#몸
mp_pose = mp.solutions.pose


# 얼굴 
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_connections = mp.solutions.face_mesh_connections


# 손
mp_hands = mp.solutions.hands

# 손가락
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS  # 손가락 뼈대 연결 정보 (시작 관절, 끝관절)


#얼굴 
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


# 손 
def draw_hands(image, results, point_radius=4, line_thickness=2):
     
    if not results or not results.multi_hand_landmarks: #객체가 없거나 손을 못찾은 경우
        return image

    h, w = image.shape[:2] # 영상에서 높이와 너비를  가져와 저장. (x,y)

    for landmarks in results.multi_hand_landmarks:
        
        # 1) 선
        for start, end in HAND_CONNECTIONS:
            x1= int(landmarks.landmark[start].x * w)
            y1 = int(landmarks.landmark[start].y * h)
            
            x2 = int(landmarks.landmark[end].x * w)
            y2 = int(landmarks.landmark[end].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)  # BGR: 초록

        # 2) 포인트
        for lm in landmarks.landmark:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(image, (cx, cy), point_radius, (0, 0, 255), -1)        # BGR: 빨강

    return image







#얼굴 
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



# 몸
def draw_pose(image, results, line_thickness=2):

    
    if not results or not results.pose_landmarks:
        return image

    h, w = image.shape[:2] #0 : 너비 , 1: 높이 
    landmarks = results.pose_landmarks.landmark

    def to_xy(idx): # 포즈 좌표 -> 픽셀로 변경
        lm = landmarks[idx] # 전체 랜드마크 배열에서 해당 인덱스의 3D 포인트를 꺼냄. 
        return int(lm.x * w), int(lm.y * h)

    orange = (0, 140, 255)  # BGR 오렌지 -> 동일한 색상 사용. 

    # 0~32에 대해 픽셀 좌표를 계산. (다 사용하진 않음)
    points ={}
    for idx in range(33):
        points[idx]=to_xy(idx)

    # 기본 포즈 라인(얼굴/손 포함) - 포인트 0~24만 사용, 11-12는 직접 연결(어깨 부분임.)
    skeleton_pairs = []
        
    for start, end in mp_pose.POSE_CONNECTIONS:
        
        if max(start, end) >= 23:
            continue
        if {start, end} in ({0, 11}, {0, 12}):#0,11 & 0&12는 
            continue
        skeleton_pairs.append((start,end))
                            
                    
    
    if (11, 12) not in skeleton_pairs and (12, 11) not in skeleton_pairs:
        skeleton_pairs.append((11, 12))

    drawn_pairs = set()
    for start, end in skeleton_pairs:
        pair_key = tuple(sorted((start, end))) #(11,12)와 (12,11)을 구분하기 위해서 
        
        if pair_key in drawn_pairs:#똑같은 선을 다시 그리지 않기 위해서 
            continue
        drawn_pairs.add(pair_key)
        cv2.line(image, points[start], points[end], orange, line_thickness)

    # 목에서 발까지 수직 라인
    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

    neck_center = midpoint(points[11], points[12])
    hip_center = midpoint(points[23], points[24])
    #knee_center = midpoint(points[25], points[26])
    #ankle_center = midpoint(points[27], points[28])
    #heel_center = midpoint(points[29], points[30])
    #foot_center = midpoint(points[31], points[32])

    central_chain = []
    central_chain.append((points[0], neck_center))
    central_chain.append((neck_center, hip_center))
    #central_chain.append((hip_center, knee_center))
    #central_chain.append((knee_center, ankle_center))
    #central_chain.append((ankle_center, heel_center))
    #central_chain.append((heel_center, foot_center))

    for start, end in central_chain:
        cv2.line(image, start, end, orange, line_thickness)

    return image





def Vedio_origin():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    with mp_pose.Pose(
        static_image_mode=False,  # F가 영상 , T가 사진
        model_complexity=1,  # 아직 2가 지원되지 않음. 해봤는데 에러남.
        enable_segmentation=False,  # F 사람 /  T 사람 + 사람 실루엣
        min_detection_confidence=0.5,  # 손바닥 탐지
        min_tracking_confidence=0.5,  # 손 추적
    ) as pose, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=True,  # F 기본적인 지점 468개 | T 홍채+ 입주변 총 478개 제공
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        # 위 설정값들은 실시간 추적에 적당한 기본 민감도로 조정한다.
    ) as face_mesh, mp_hands.Hands(
        static_image_mode=False,  # false면 영상 , true는 이미지.
        max_num_hands=8,  # 양손
        model_complexity=1,  # 모델의 복잡도 : 0은 빠르지만 정확도 떨어짐 ~2로 갈 수록 정밀해짐.
        min_detection_confidence=0.5,  # 손바닥 감지할때 쓰이는 가중치
        min_tracking_confidence=0.5,  # 손 추척이 성공으로 간주되는  가중치
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            #셀카처럼 보이게 좌우 반전 
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(rgb)
            face_results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)
            

            #한번에 보이게 하려면 프레임을 카피함 -> 그 위에 순서대로 덮어 씀. 
            overlay_frame = frame.copy()

            draw_pose(overlay_frame, pose_results, line_thickness=2)
            draw_face(overlay_frame, face_results, line_thickness=2)
            draw_hands(
                overlay_frame, hand_results, point_radius=4, line_thickness=2
            )

            cv2.imshow("Full Body Overlay", overlay_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"): #종료조건 2
                break

    cap.release()
    cv2.destroyAllWindows() #gui 종료 


if __name__ == "__main__":
    Vedio_origin()


