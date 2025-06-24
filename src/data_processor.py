import os
import pickle

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from config import (
    IMAGE_FILE_EXTENSIONS,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    RAW_DATA_DIR,
    SEQUENCE_LENGTH,
    VIDEO_FILE_EXTENSIONS,
)

mp_hands = mp.solutions.hands


def extract_keypoints_from_image(image_rgb, hands_instance):
    results = hands_instance.process(image_rgb)
    hand = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        hand = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return hand


def process_and_extract_sequences():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Directory not found: {RAW_DATA_DIR}")
        os.makedirs(RAW_DATA_DIR)
        return

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    actions = [
        d
        for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ]
    action_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        for action in tqdm(actions, desc="Processing actions"):
            action_path = os.path.join(RAW_DATA_DIR, action)

            for file_name in os.listdir(action_path):
                file_path = os.path.join(action_path, file_name)

                if file_name.lower().endswith(VIDEO_FILE_EXTENSIONS):
                    cap = cv2.VideoCapture(file_path)
                    keypoints_video = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        keypoints = extract_keypoints_from_image(image_rgb, hands)
                        keypoints_video.append(keypoints)
                    cap.release()

                    num_sequences = len(keypoints_video) // SEQUENCE_LENGTH
                    for i in range(num_sequences):
                        sequences.append(
                            keypoints_video[
                                i * SEQUENCE_LENGTH : (i + 1) * SEQUENCE_LENGTH
                            ]
                        )
                        labels.append(action_map[action])

                elif file_name.lower().endswith(IMAGE_FILE_EXTENSIONS):
                    frame = cv2.imread(file_path)
                    if frame is None:
                        continue
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    keypoints = extract_keypoints_from_image(image_rgb, hands)

                    if np.sum(keypoints) != 0:
                        static_sequence = [keypoints] * SEQUENCE_LENGTH
                        sequences.append(static_sequence)
                        labels.append(action_map[action])

    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump(
            {
                "sequences": np.array(sequences),
                "labels": np.array(labels),
                "actions": actions,
            },
            f,
        )

    print(f"\nData processing complete. Saved to {PROCESSED_DATA_FILE}")
    print(f"Total sequences created: {len(sequences)}")
    print(f"Number of classes: {len(actions)}")


if __name__ == "__main__":
    process_and_extract_sequences()
