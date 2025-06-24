import os
import pickle

import cv2
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm

from config import (
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    RAW_VIDEO_DIR,
    SEQUENCE_LENGTH,
    VIDEO_FILE_EXTENSIONS,
)

mp_hands = mp.solutions.hands


def extract_keypoints(results):
    hand = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        hand = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return hand


def process_and_extract_sequences():
    if not os.path.exists(RAW_VIDEO_DIR):
        print(f"Directory not found: {RAW_VIDEO_DIR}")
        os.makedirs(RAW_VIDEO_DIR)
        return

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    actions = [
        d
        for d in os.listdir(RAW_VIDEO_DIR)
        if os.path.isdir(os.path.join(RAW_VIDEO_DIR, d))
    ]
    action_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        for action in tqdm(actions, desc="Processing actions"):
            action_path = os.path.join(RAW_VIDEO_DIR, action)
            video_files = [
                f for f in os.listdir(action_path) if f.endswith(VIDEO_FILE_EXTENSIONS)
            ]

            for video_file in video_files:
                video_path = os.path.join(action_path, video_file)
                cap = cv2.VideoCapture(video_path)

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count < 1:
                    cap.release()
                    continue

                keypoints_video = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)
                    keypoints = extract_keypoints(results)
                    keypoints_video.append(keypoints)
                cap.release()

                if len(keypoints_video) > 0:
                    num_sequences = len(keypoints_video) // SEQUENCE_LENGTH
                    for i in range(num_sequences):
                        start = i * SEQUENCE_LENGTH
                        end = start + SEQUENCE_LENGTH
                        sequences.append(keypoints_video[start:end])
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
