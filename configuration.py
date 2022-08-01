import os
from math import ceil

SEED = 47

DATA_FOLDERS = [os.sep.join(["datasets", "rpg-maker-xp"])]

DIRECTION_FOLDERS = ["0-back", "1-left", "2-front", "3-right"]
DATASET_SIZES = [294]

DATASET_SIZE = sum(DATASET_SIZES)
TRAIN_PERCENTAGE = 0.85
TRAIN_SIZES = [ceil(n * TRAIN_PERCENTAGE) for n in DATASET_SIZES]
TRAIN_SIZE = sum(TRAIN_SIZES)
TEST_SIZES = [DATASET_SIZES[i] - TRAIN_SIZES[i] for i, n in enumerate(DATASET_SIZES)]
TEST_SIZE = sum(TEST_SIZES)

BUFFER_SIZE = DATASET_SIZE
BATCH_SIZE = 4

IMG_SIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

# for indexed colors
MAX_PALETTE_SIZE = 256
INVALID_INDEX_COLOR = [255, 0, 220, 255]    # some pink

TEMP_FOLDER = "temp-side2side"

