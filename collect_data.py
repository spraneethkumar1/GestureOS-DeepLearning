"""
GestureOS - Dataset collector
Run: python collect_data.py
Controls:
  1 = Open_Palm, 2 = Fist, 3 = Peace, 4 = Thumbs_Up, 5 = Stop
  SPACE = save one landmark sample for the currently selected class
  Q = quit
"""
import csv
import os
import cv2
import mediapipe as mp
import numpy as np

DATA_DIR = "dataset"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
LABELS = {
    1: "Open_Palm",
    2: "Fist",
    3: "Peace",
    4: "Thumbs_Up",
    5: "Stop",
}
HEADER = ["label"] + [f"f{i}" for i in range(63)]

def normalize_landmarks(hand_landmarks):
    """Return 63 wrist-relative, scale-normalized x/y/z values."""
    points = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    points -= points[0]  # wrist-relative coordinates
    scale = float(np.max(np.abs(points[:, :2])))
    if scale > 1e-6:
        points /= scale
    return points.flatten().tolist()

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    if not file_exists:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    selected = None
    saved_count = 0

    # Count existing samples for display.
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            saved_count = max(0, sum(1 for _ in csv.reader(f)) - 1)
    except OSError:
        pass

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Close other camera apps and check Windows camera permissions.")
        return

    print("Choose a class with keys 1-5, then press SPACE to save a sample. Press Q to quit.")
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
                print("Could not read a frame from the webcam.")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            landmarks = None

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                landmarks = normalize_landmarks(hand)

            title = f"Selected: {selected or 'None'} | Saved total: {saved_count}"
            cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            cv2.putText(frame, "1 Palm  2 Fist  3 Peace  4 ThumbsUp  5 Stop", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "SPACE=save sample | Q=quit", (10, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            if landmarks is None:
                cv2.putText(frame, "No hand detected", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.imshow("GestureOS - Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord(str(i)) for i in range(1, 6)]:
                selected = LABELS[int(chr(key))]
                print("Selected class:", selected)
            elif key == 32:  # space
                if selected is None:
                    print("Select a class first using keys 1-5.")
                elif landmarks is None:
                    print("No hand detected. Keep your hand visible and try again.")
                else:
                    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([selected] + landmarks)
                    saved_count += 1
                    print(f"Saved {selected} sample. Total samples: {saved_count}")
            elif key in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
