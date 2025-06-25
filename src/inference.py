import pickle

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from config import *
from model import QuantizedSignLanguageTransformer, SignLanguageTransformer


def draw_wrapped_text(canvas, text, org, font, scale, color, thickness, max_width):
    """
    Função para desenhar texto com quebra de linha automática dentro de uma largura máxima.
    """
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        (text_width, _), _ = cv2.getTextSize(test_line, font, scale, thickness)

        if text_width > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    lines.append(current_line)

    x, y = org
    for i, line in enumerate(lines):
        line_y = y + i * int(scale * 40)
        cv2.putText(
            canvas, line, (x, line_y), font, scale, color, thickness, cv2.LINE_AA
        )


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

    float_model = SignLanguageTransformer(
        input_size=INPUT_SIZE,
        d_model=HIDDEN_UNITS,
        nhead=NUM_HEADS,
        num_encoder_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    )
    quantized_model = QuantizedSignLanguageTransformer(float_model)

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
    sentence_str = ""
    last_predicted_action = None

    CAM_WIDTH, CAM_HEIGHT = 640, 480
    UI_WIDTH = 400
    CANVAS_WIDTH = CAM_WIDTH + UI_WIDTH
    CANVAS_HEIGHT = CAM_HEIGHT

    BG_COLOR = (48, 48, 48)
    TEXT_AREA_COLOR = (30, 30, 30)
    TEXT_COLOR = (240, 240, 240)
    PREDICTION_BG_COLOR = (80, 80, 80)

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BG_COLOR, dtype=np.uint8)
        canvas[0:CAM_HEIGHT, 0:CAM_WIDTH] = frame

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands_instance.process(image_rgb)

        current_prediction_text = "Nenhuma mao detectada"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(
                canvas[0:CAM_HEIGHT, 0:CAM_WIDTH],
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
            )
            keypoints = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            ).flatten()
            sequence_data.append(keypoints)
            sequence_data = sequence_data[-SEQUENCE_LENGTH:]

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
                            predicted_action = actions[
                                prediction_idx
                            ].lower()  # Converte para minúsculo
                            current_prediction_text = (
                                f"{predicted_action.upper()} ({confidence:.2%})"
                            )

                            if predicted_action != last_predicted_action:
                                # --- LÓGICA DE COMANDOS ESPECIAIS ---
                                if predicted_action == "space":
                                    sentence_str += " "
                                elif predicted_action == "del":
                                    sentence_str = sentence_str[:-1]
                                else:
                                    sentence_str += predicted_action.upper()
                                # ------------------------------------
                                last_predicted_action = predicted_action
                        else:
                            last_predicted_action = None
                            current_prediction_text = "..."
                except Exception as e:
                    current_prediction_text = "Erro"
                    print(e)
            else:
                current_prediction_text = "COLETANDO..."

        cv2.rectangle(
            canvas, (CAM_WIDTH, 0), (CANVAS_WIDTH, CANVAS_HEIGHT), TEXT_AREA_COLOR, -1
        )
        cv2.putText(
            canvas,
            "TEXTO RECONHECIDO",
            (CAM_WIDTH + 20, 40),
            font_face,
            font_scale,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.line(canvas, (CAM_WIDTH + 20, 50), (CANVAS_WIDTH - 20, 50), TEXT_COLOR, 1)

        draw_wrapped_text(
            canvas,
            sentence_str,
            (CAM_WIDTH + 25, 90),
            font_face,
            font_scale,
            TEXT_COLOR,
            1,
            UI_WIDTH - 40,
        )

        cv2.rectangle(canvas, (0, 0), (CAM_WIDTH, 40), PREDICTION_BG_COLOR, -1)
        cv2.putText(
            canvas,
            current_prediction_text,
            (10, 30),
            font_face,
            1,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("QPruner-Libras - TCC", canvas)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

        # A lógica de teclado para edição manual pode ser mantida ou removida,
        # já que agora temos os gestos.
        if key == 32:  # Tecla Espaço manual
            sentence_str += " "
            last_predicted_action = None
        if key == 8:  # Tecla Backspace manual
            sentence_str = sentence_str[:-1]
            last_predicted_action = None

    cap.release()
    cv2.destroyAllWindows()
    mp_hands_instance.close()


if __name__ == "__main__":
    main()
