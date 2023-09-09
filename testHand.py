import cv2
import mediapipe as mp

# path
input_image_path = 'testImages/14845.jpg'
output_image_path = 'testImages/14845_output.jpg'

# 初期化部
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence=0.5
    )
keyPoint_coordinate = []

# 画像のインポート
image = cv2.imread(input_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# キーポイントの推定
results = hands.process(image_rgb)

# キーポイントと骨格の描画
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            heigth, width, _ = image.shape
            cx, cy = int(landmark.x * width), int(landmark.y * heigth)
            cv2.circle(image, (cx, cy), 10, (255, 0, 0), -1)
            keyPoint_coordinate.append([cx, cy])
            
        for connection in mp_hands.HAND_CONNECTIONS:
            x0, y0 = int(hand_landmarks.landmark[connection[0]].x * width), int(hand_landmarks.landmark[connection[0]].y * heigth)
            x1, y1 = int(hand_landmarks.landmark[connection[1]].x * width), int(hand_landmarks.landmark[connection[1]].y * heigth)
            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 3, lineType=cv2.LINE_AA)

# 出力
cv2.imwrite(output_image_path, image)
print(keyPoint_coordinate)

hands.close()