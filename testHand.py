import cv2
import mediapipe as mp
import os

def make_keypoints(input_dir_path, output_dir_path, start_number, finish_number):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    for i in range(start_number, finish_number + 1):
        # ファイルパスの設定
        input_image_path = f'{input_dir_path}/{i}.jpg'
        output_image_path = f'{output_dir_path}/{i}_output.jpg'

        # 画像の読み込み
        image = cv2.imread(input_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # キーポイントの推定
        results = hands.process(image_rgb)

        # キーポイントと骨格の描画
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    height, width, _ = image.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(image, (cx, cy), 10, (255, 0, 0), -1)

            for connection in mp_hands.HAND_CONNECTIONS:
                x0, y0 = int(hand_landmarks.landmark[connection[0]].x * width), int(hand_landmarks.landmark[connection[0]].y * height)
                x1, y1 = int(hand_landmarks.landmark[connection[1]].x * width), int(hand_landmarks.landmark[connection[1]].y * height)
                cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 3, lineType=cv2.LINE_AA)

        # 出力
        cv2.imwrite(output_image_path, image)

    hands.close()

make_keypoints('images/0001', 'images_result/0001', 12313, 12360)
