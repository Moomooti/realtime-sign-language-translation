import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS  # 손가락 뼈대 연결 정보 (시작 관절, 끝관절)

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



def Video_origin():

    cap = cv2.VideoCapture(0)  # 웹캠
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # MediaPipe Hands 초기화
    #with  A() as obj : A()가 실행될때 obj를 만들고 블록을 빠져나올때 자동으로 작업을 해줌 : 자원관리
    with mp_hands.Hands(
        static_image_mode=False, #false면 영상 , Ture는 이미지.
        max_num_hands=2, #양손
        model_complexity=1, #모델의 복잡도 : 0은 빠르지만 정확도 떨어짐 ~2로 갈 수록 정밀해짐. 
        min_detection_confidence=0.5, # 손바닥 감지할때 쓰이는 가중치 
        min_tracking_confidence=0.5 #손 추척이 성공으로 간주되는  가중치 
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret: #종료 조건 1 : 프레임을 못읽어 올때 
                break

            # 셀카처럼 보이게 좌우 반전
            frame = cv2.flip(frame, 1)

            # MediaPipe 입력은 RGB 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # 포인트/선 그리기
            out = draw_hands(frame, results,point_radius=4, line_thickness=2)

            cv2.imshow("Hands (points: red, lines: green)", out)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): #종료 조건 2 : q를 눌렀을때.
                break

            

    cap.release()
    cv2.destroyAllWindows() #종료 조건3 : gui 창을 닫을떄 


if __name__ == "__main__":
    Video_origin()
