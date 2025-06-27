import glob
import json
import os
import pickle

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.config import *

# CORREÇÃO: Importa a função builder em vez da classe antiga
from src.model import SignLanguageTransformer, build_quantized_model


class InferenceApp:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1
        )
        self.actions, self.num_classes = self._load_actions()
        self.available_models = self._load_all_models()
        self.model_names = list(self.available_models.keys())
        self.current_model_name = self.model_names[0] if self.model_names else None
        self.sequence_data = []
        self.sentence_str = ""
        self.last_predicted_action = None
        self._setup_ui_constants()

    def _setup_ui_constants(self):
        self.CAM_WIDTH, self.CAM_HEIGHT = 640, 480
        self.UI_WIDTH = 400
        self.CANVAS_WIDTH = self.CAM_WIDTH + self.UI_WIDTH
        self.CANVAS_HEIGHT = self.CAM_HEIGHT
        self.BG_COLOR = (48, 48, 48)
        self.TEXT_AREA_COLOR = (30, 30, 30)
        self.TEXT_COLOR = (240, 240, 240)
        self.HIGHLIGHT_COLOR = (0, 255, 255)
        self.PREDICTION_BG_COLOR = (80, 80, 80)
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.8
        self.SMALL_FONT_SCALE = 0.6

    def _load_actions(self):
        try:
            with open(PROCESSED_DATA_FILE, "rb") as f:
                data = pickle.load(f)
            return data["actions"], len(data["actions"])
        except FileNotFoundError:
            print(
                f"Erro: Arquivo de dados processados não encontrado em {PROCESSED_DATA_FILE}."
            )
            return None, 0

    def _load_all_models(self):
        if not self.num_classes:
            return {}

        models = {}
        model_paths = glob.glob(os.path.join(str(MODEL_SAVE_DIR), "*.pth"))
        results_path = os.path.join(str(RESULTS_DIR), "experiment_results.json")

        try:
            with open(results_path, "r") as f:
                results_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results_data = {}
            print(
                f"Aviso: Arquivo de resultados não encontrado ou inválido. Acurácia não será exibida."
            )

        for path in model_paths:
            model_name = os.path.basename(path).replace(".pth", "")

            # Usamos a mesma classe base SignLanguageTransformer para carregar
            # o state_dict, pois a poda não altera a definição da classe.
            float_model = SignLanguageTransformer(
                num_classes=self.num_classes,
                input_size=INPUT_SIZE,
                d_model=HIDDEN_UNITS,
                nhead=NUM_HEADS,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
            )
            float_model.load_state_dict(torch.load(path, map_location="cpu"))
            float_model.eval()

            # CORREÇÃO: Usa a função builder para criar a versão quantizada
            quantized_version = build_quantized_model(float_model)

            model_stats = results_data.get(model_name, {})
            models[model_name] = {
                "model": quantized_version,
                "size_mb": os.path.getsize(path) / 1e6,
                "accuracy": model_stats.get("accuracy", 0.0),
            }

        return {name: models[name] for name in sorted(models.keys())}

    def _draw_wrapped_text(self, canvas, text, org, max_width):
        words = text.split(" ")
        lines, current_line = [], ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            (text_width, _), _ = cv2.getTextSize(
                test_line, self.FONT_FACE, self.FONT_SCALE, 1
            )
            if text_width > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        lines.append(current_line)
        x, y = org
        for i, line in enumerate(lines):
            line_y = y + i * int(self.FONT_SCALE * 40)
            cv2.putText(
                canvas,
                line,
                (x, line_y),
                self.FONT_FACE,
                self.FONT_SCALE,
                self.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

    def _handle_input(self, key):
        if key == ord("q"):
            return False
        if key == ord("m") and self.model_names:
            current_idx = self.model_names.index(self.current_model_name)
            next_idx = (current_idx + 1) % len(self.model_names)
            self.current_model_name = self.model_names[next_idx]
        if key == 32:
            self.sentence_str += " "
            self.last_predicted_action = None
        if key == 8:
            self.sentence_str = self.sentence_str[:-1]
            self.last_predicted_action = None
        return True

    def _update_ui(self, canvas, current_prediction_text):
        cv2.rectangle(
            canvas,
            (self.CAM_WIDTH, 0),
            (self.CANVAS_WIDTH, self.CANVAS_HEIGHT),
            self.TEXT_AREA_COLOR,
            -1,
        )

        active_model_info = self.available_models[self.current_model_name]
        cv2.putText(
            canvas,
            "MODELO ATIVO (press 'm')",
            (self.CAM_WIDTH + 20, 40),
            self.FONT_FACE,
            0.7,
            self.TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            self.current_model_name,
            (self.CAM_WIDTH + 25, 70),
            self.FONT_FACE,
            self.SMALL_FONT_SCALE,
            self.HIGHLIGHT_COLOR,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Tamanho: {active_model_info.get('size_mb', 0):.2f} MB",
            (self.CAM_WIDTH + 25, 100),
            self.FONT_FACE,
            self.SMALL_FONT_SCALE,
            self.TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Acuracia: {active_model_info.get('accuracy', 0):.2%}",
            (self.CAM_WIDTH + 25, 120),
            self.FONT_FACE,
            self.SMALL_FONT_SCALE,
            self.TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

        cv2.line(
            canvas, (CAM_WIDTH + 20, 140), (CANVAS_WIDTH - 20, 140), self.TEXT_COLOR, 1
        )

        self._draw_wrapped_text(
            canvas, self.sentence_str, (self.CAM_WIDTH + 25, 170), self.UI_WIDTH - 40
        )

        cv2.rectangle(
            canvas, (0, 0), (self.CAM_WIDTH, 40), self.PREDICTION_BG_COLOR, -1
        )
        cv2.putText(
            canvas,
            current_prediction_text,
            (10, 30),
            self.FONT_FACE,
            1,
            self.TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    def run(self):
        if not self.actions or not self.available_models:
            print("Encerrando: Ações ou modelos não puderam ser carregados.")
            return

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_HEIGHT)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            canvas = np.full(
                (self.CANVAS_HEIGHT, self.CANVAS_WIDTH, 3),
                self.BG_COLOR,
                dtype=np.uint8,
            )
            canvas[0 : self.CAM_HEIGHT, 0 : self.CAM_WIDTH] = frame

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            current_prediction_text = "No hand detected"
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(
                    canvas[0 : self.CAM_HEIGHT, 0 : self.CAM_WIDTH],
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )
                keypoints = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                ).flatten()
                self.sequence_data.append(keypoints)
                self.sequence_data = self.sequence_data[-SEQUENCE_LENGTH:]

                if len(self.sequence_data) == SEQUENCE_LENGTH:
                    active_model = self.available_models[self.current_model_name][
                        "model"
                    ]
                    input_tensor = torch.tensor(
                        np.array([self.sequence_data]), dtype=torch.float32
                    )
                    with torch.no_grad():
                        output = active_model(input_tensor)
                        confidence = torch.softmax(output, dim=1).max().item()

                        if confidence > INFERENCE_CONFIDENCE_THRESHOLD:
                            pred_idx = torch.argmax(output, dim=1).item()
                            pred_action = self.actions[pred_idx].lower()
                            current_prediction_text = (
                                f"{pred_action.upper()} ({confidence:.2%})"
                            )
                            if pred_action != self.last_predicted_action:
                                if pred_action == "space":
                                    self.sentence_str += " "
                                elif pred_action == "del":
                                    self.sentence_str = self.sentence_str[:-1]
                                else:
                                    self.sentence_str += pred_action.upper()
                                self.last_predicted_action = pred_action
                        else:
                            self.last_predicted_action = None
                            current_prediction_text = "..."
                else:
                    current_prediction_text = "COLLECTING..."

            self._update_ui(canvas, current_prediction_text)
            cv2.imshow("QPruner-Libras - TCC", canvas)

            if not self._handle_input(cv2.waitKey(10) & 0xFF):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
