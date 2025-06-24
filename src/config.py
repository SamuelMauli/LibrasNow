import os

import torch

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw_videos")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "sequences_and_labels.pkl")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "..", "models")

# --- Data Processing ---
SEQUENCE_LENGTH = 30
VIDEO_FILE_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# --- Model Architecture ---
INPUT_SIZE = 21 * 3
HIDDEN_UNITS = 256
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.2

# --- Training ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
SAM_RHO = 0.05
EARLY_STOPPING_PATIENCE = 10

# --- Compression ---
PRUNING_AMOUNT = 0.4

# --- WandB Logging ---
WANDB_PROJECT = "QPruner-Libras"
