import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
web_came = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while web_came.isOpened():
        ret, frame = web_came.read()
        # convert to BGR2RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        results = holistic.process(image)
        # convert back to RGB2BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Landmarks
        # Right_hand_landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(0, 200, 130), thickness=2, circle_radius=5))
        #left_hand_landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(20,255, 5), thickness=2, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(4, 10, 250), thickness=2, circle_radius=5))

        cv2.imshow("Video", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

web_came.release()
cv2.destroyAllWindows()