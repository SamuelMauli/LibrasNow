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
        print(
            f"Processed data file not found at {PROCESSED_DATA_FILE}. Cannot load class names."
        )
        return

    model = SignLanguageTransformer(
        input_size=INPUT_SIZE,
        d_model=HIDDEN_UNITS,
        nhead=NUM_HEADS,
        num_encoder_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    )

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    compressed_model_path = os.path.join(
        MODEL_SAVE_DIR, "compressed_model_quantized.pth"
    )
    if not os.path.exists(compressed_model_path):
        print(
            f"Compressed model not found at {compressed_model_path}. Please run compress.py first."
        )
        return

    quantized_model.load_state_dict(torch.load(compressed_model_path))
    quantized_model.eval()

    print("Compressed model loaded for inference. Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    mp_hands_instance = mp.solutions.hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1
    )

    sequence_data = []
    current_prediction_text = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands_instance.process(image_rgb)

        keypoints = np.zeros(21 * 3)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )
            keypoints = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            ).flatten()

        sequence_data.append(keypoints)
        sequence_data = sequence_data[-SEQUENCE_LENGTH:]

        display_text = "COLETANDO..."
        if len(sequence_data) == SEQUENCE_LENGTH:
            try:
                input_tensor = torch.tensor(
                    np.array([sequence_data]), dtype=torch.float32
                )
                with torch.no_grad():
                    output = quantized_model(input_tensor)
                    confidence = torch.softmax(output, dim=1).max().item()

                    if confidence > INFERENCE_CONFIDENCE_THRESHOLD:
                        prediction_idx = torch.argmax(output, dim=1).item()
                        current_prediction_text = (
                            f"{actions[prediction_idx]} ({confidence:.2%})"
                        )

                display_text = current_prediction_text
            except Exception as e:
                display_text = "Erro na inferencia"
                print(e)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(
            frame,
            display_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("QPruner-Libras Inference", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_hands_instance.close()


if __name__ == "__main__":
    main()
