# src/data_processing.py (VERSÃO CORRIGIDA)

import os
import pickle

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from config import (
    AUGMENTATION_PROB,
    IMAGE_FILE_EXTENSIONS,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    RAW_DATA_DIR,
    SEQUENCE_LENGTH,
    VIDEO_FILE_EXTENSIONS,
)

mp_hands = mp.solutions.hands

augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(p=0.2),
    ],
    p=AUGMENTATION_PROB,
)


def extract_keypoints_from_image(image_rgb, hands_instance):
    results = hands_instance.process(image_rgb)
    keypoints = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return keypoints


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    actions = [
        d
        for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ]
    action_map = {label: num for num, label in enumerate(actions)}

    all_sequences = []

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        for action in tqdm(actions, desc="Processing actions"):
            action_path = os.path.join(RAW_DATA_DIR, action)
            file_list = os.listdir(action_path)

            if not file_list:
                print(f"\nAVISO: A pasta da classe '{action}' está vazia. Pulando.")
                continue

            for file_name in file_list:
                file_path = os.path.join(action_path, file_name)

                # --- LÓGICA CORRIGIDA PARA VÍDEOS E IMAGENS ---
                if file_name.lower().endswith(VIDEO_FILE_EXTENSIONS):
                    cap = cv2.VideoCapture(file_path)
                    keypoints_video = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        augmented = augmentation_pipeline(image=frame)
                        image_rgb = cv2.cvtColor(augmented["image"], cv2.COLOR_BGR2RGB)
                        keypoints = extract_keypoints_from_image(image_rgb, hands)
                        keypoints_video.append(keypoints)
                    cap.release()

                    # CORREÇÃO: Usa >= em vez de >
                    if len(keypoints_video) >= SEQUENCE_LENGTH:
                        num_sequences = len(keypoints_video) // SEQUENCE_LENGTH
                        for i in range(num_sequences):
                            all_sequences.append(
                                {
                                    "sequence": keypoints_video[
                                        i * SEQUENCE_LENGTH : (i + 1) * SEQUENCE_LENGTH
                                    ],
                                    "label": action_map[action],
                                }
                            )

                elif file_name.lower().endswith(IMAGE_FILE_EXTENSIONS):
                    frame = cv2.imread(file_path)
                    if frame is not None:
                        augmented = augmentation_pipeline(image=frame)
                        image_rgb = cv2.cvtColor(augmented["image"], cv2.COLOR_BGR2RGB)
                        keypoints = extract_keypoints_from_image(image_rgb, hands)

                        # Se uma mão foi detectada na imagem estática
                        if np.sum(keypoints) != 0:
                            # Cria uma sequência completa repetindo a pose estática
                            static_sequence = [keypoints] * SEQUENCE_LENGTH
                            all_sequences.append(
                                {
                                    "sequence": static_sequence,
                                    "label": action_map[action],
                                }
                            )

    if not all_sequences:
        print("\n!!!!!! ERRO CRÍTICO NO PROCESSAMENTO DE DADOS !!!!!!")
        print("Nenhuma sequência foi criada. Verifique se:")
        print("1. A pasta 'data/raw_data/' contém subpastas com seus vídeos/imagens.")
        print("2. Os vídeos têm pelo menos 30 frames.")
        print("3. As mãos estão claramente visíveis nos seus dados.")
        return  # Impede a criação de um arquivo vazio

    sequences_np = np.array([item["sequence"] for item in all_sequences])
    labels_np = np.array([item["label"] for item in all_sequences])

    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump(
            {"sequences": sequences_np, "labels": labels_np, "actions": actions}, f
        )

    print(f"\nData processing complete. Saved to {PROCESSED_DATA_FILE}")
    print(f"Total sequences created: {len(sequences_np)}")


if __name__ == "__main__":
    main()
