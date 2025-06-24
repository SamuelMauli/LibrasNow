import pickle

import cv2
import mediapipe as mp
import numpy as np
import torch

from config import *
from model import SignLanguageTransformer


def main():
    try:
        with open(PROCESSED_DATA_FILE, "rb") as f:
            data = pickle.load(f)
        actions = data["actions"]
        num_classes = len(actions)
    except FileNotFoundError:
        print("Processed data file not found. Cannot load class names.")
        return

    # --- Load Compressed Model ---
    model = SignLanguageTransformer(
        input_size=INPUT_SIZE,
        d_model=HIDDEN_UNITS,
        nhead=NUM_HEADS,
        num_encoder_layers=NUM_LAYERS,
        num_classes=num_classes,
    )

    # We must quantize the model architecture before loading the state_dict
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    compressed_model_path = os.path.join(
        MODEL_SAVE_DIR, "compressed_model_quantized.pth"
    )
    if not os.path.exists(compressed_model_path):
        print("Compressed model not found. Please run compress.py first.")
        return

    quantized_model.load_state_dict(torch.load(compressed_model_path))
    quantized_model.eval()

    print("Compressed model loaded for inference.")

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1
    )

    sequence = []
    current_prediction = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)

        keypoints = np.zeros(21 * 3)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )
            keypoints = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            ).flatten()

        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        if len(sequence) == SEQUENCE_LENGTH:
            input_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32)
            with torch.no_grad():
                output = quantized_model(input_tensor)
                confidence = torch.softmax(output, dim=1).max().item()
                if confidence > 0.8:
                    prediction_idx = torch.argmax(output, dim=1).item()
                    current_prediction = actions[prediction_idx]

        cv2.putText(
            frame,
            current_prediction,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
