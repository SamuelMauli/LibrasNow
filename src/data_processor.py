import pickle
from pathlib import Path

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from src.config import (
    IMAGE_FILE_EXTENSIONS,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    RAW_DATA_DIR,
    SEQUENCE_LENGTH,
    VIDEO_FILE_EXTENSIONS,
)


class DataProcessor:
    """
    Encapsula toda a lógica de processamento de dados, desde a leitura de
    arquivos brutos até a criação de um arquivo de sequências pronto para o treinamento.
    """

    def __init__(self, augment_prob: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.hands_instance = self.mp_hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
        )
        self.augmentation_pipeline = self._build_augmentation_pipeline(augment_prob)

    def _build_augmentation_pipeline(self, probability: float):
        """
        # Fundamento Acadêmico: Data Augmentation
        # Referência: Shorten & Khoshgoftaar, "A survey on Image Data Augmentation for
        # Deep Learning" (2019).
        # O que faz: Cria um pipeline de aumento de dados para gerar variações sintéticas
        # das imagens de entrada. Isso ajuda a rede neural a generalizar melhor,
        # tornando-a mais robusta a diferentes condições de iluminação, orientação e
        # pequenas oclusões, um fator crucial para a robustez de um sistema de TCC.
        """
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.ColorJitter(p=0.3),
                A.GaussNoise(p=0.3),
                A.MotionBlur(p=0.2),
            ],
            p=probability,
        )

    def _extract_keypoints(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extrai os keypoints da mão de uma única imagem RGB."""
        results = self.hands_instance.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return np.zeros(21 * 3)

    def _process_video_file(self, file_path: Path) -> list:
        """Processa um arquivo de vídeo, extraindo keypoints de cada frame."""
        cap = cv2.VideoCapture(str(file_path))
        keypoints_video = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            augmented = self.augmentation_pipeline(image=frame)
            image_rgb = cv2.cvtColor(augmented["image"], cv2.COLOR_BGR2RGB)
            keypoints = self._extract_keypoints(image_rgb)
            keypoints_video.append(keypoints)
        cap.release()
        return keypoints_video

    def _process_image_file(self, file_path: Path) -> list:
        """Processa uma imagem estática, duplicando-a para formar uma sequência."""
        frame = cv2.imread(str(file_path))
        if frame is None:
            return []
        augmented = self.augmentation_pipeline(image=frame)
        image_rgb = cv2.cvtColor(augmented["image"], cv2.COLOR_BGR2RGB)
        keypoints = self._extract_keypoints(image_rgb)
        if np.sum(keypoints) == 0:
            return []
        return [keypoints] * SEQUENCE_LENGTH

    def process(self):
        """
        Orquestra o processo completo de extração de dados.
        """
        print("Iniciando processamento de dados...")
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        actions = [d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
        action_map = {label: num for num, label in enumerate(actions)}

        all_sequences = []
        for action in tqdm(actions, desc="Processando classes"):
            action_path = RAW_DATA_DIR / action
            for file_path in action_path.iterdir():
                keypoints_list = []
                if file_path.suffix.lower() in VIDEO_FILE_EXTENSIONS:
                    keypoints_list = self._process_video_file(file_path)
                elif file_path.suffix.lower() in IMAGE_FILE_EXTENSIONS:
                    keypoints_list = self._process_image_file(file_path)

                if len(keypoints_list) >= SEQUENCE_LENGTH:
                    for i in range(len(keypoints_list) // SEQUENCE_LENGTH):
                        start, end = i * SEQUENCE_LENGTH, (i + 1) * SEQUENCE_LENGTH
                        all_sequences.append(
                            {
                                "sequence": keypoints_list[start:end],
                                "label": action_map[action],
                            }
                        )

        if not all_sequences:
            print(
                "\nERRO CRÍTICO: Nenhuma sequência foi criada. Verifique o diretório 'data/raw'."
            )
            return

        sequences_np = np.array([item["sequence"] for item in all_sequences])
        labels_np = np.array([item["label"] for item in all_sequences])

        with open(PROCESSED_DATA_FILE, "wb") as f:
            pickle.dump(
                {"sequences": sequences_np, "labels": labels_np, "actions": actions}, f
            )

        print(
            f"\nProcessamento concluído. {len(sequences_np)} sequências salvas em '{PROCESSED_DATA_FILE}'"
        )
