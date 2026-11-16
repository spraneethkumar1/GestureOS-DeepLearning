"""
GestureOS - Real-time gesture prediction
Run after training: python gesture_recognition.py
Press Q to quit.
"""
import json
import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/gesture_model.keras"
LABELS_PATH = "models/labels.json"

def normalize_landmarks(hand_landmarks):
    points = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    points -= points[0]
    scale = float(np.max(np.abs(points[:, :2])))
    if scale > 1e-6:
        points /= scale
    return points.flatten()

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise SystemExit("Trained model not found. Run collect_data.py and train_model.py first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check Windows camera permissions and close other camera apps.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                features = normalize_landmarks(hand).reshape(1, 63)
                probabilities = model.predict(features, verbose=0)[0]
                idx = int(np.argmax(probabilities))
                confidence = float(probabilities[idx])
                label = labels[idx]
                text = f"{label}: {confidence * 100:.1f}%"
                color = (0, 220, 0) if confidence >= 0.65 else (0, 165, 255)
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                if confidence < 0.65:
                    cv2.putText(frame, "Low confidence - try another hand position",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(frame, "Show one hand to the camera", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, "GestureOS | Press Q to quit", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("GestureOS - Deep Learning Recognition", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
