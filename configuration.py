import argparse
import os
from math import ceil

SEED = 47

DATA_FOLDERS = [os.sep.join(["datasets", "rpg-maker-xp"])]

DIRECTIONS = ["back", "left", "front", "right"]
DIRECTION_BACK = DIRECTIONS.index("back")                                   # 0
DIRECTION_LEFT = DIRECTIONS.index("left")                                   # 1
DIRECTION_FRONT = DIRECTIONS.index("front")                                 # 2
DIRECTION_RIGHT = DIRECTIONS.index("right")                                 # 3
# ["0-back", "1-left", "2-front", "3-right"]
DIRECTION_FOLDERS = [f"{i}-{name}" for i, name in enumerate(DIRECTIONS)]

DATASET_SIZES = [294]
DATASET_SIZE = sum(DATASET_SIZES)
TRAIN_PERCENTAGE = 0.85
TRAIN_SIZES = [ceil(n * TRAIN_PERCENTAGE) for n in DATASET_SIZES]
TRAIN_SIZE = sum(TRAIN_SIZES)
TEST_SIZES = [DATASET_SIZES[i] - TRAIN_SIZES[i]
              for i, n in enumerate(DATASET_SIZES)]
TEST_SIZE = sum(TEST_SIZES)

BUFFER_SIZE = DATASET_SIZE
BATCH_SIZE = 4

IMG_SIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

# for indexed colors
MAX_PALETTE_SIZE = 256
INVALID_INDEX_COLOR = [255, 0, 220, 255]    # some hotpink

TEMP_FOLDER = "temp-side2side"


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class OptionParser(metaclass=SingletonMeta):
	def __init__(self) -> None:
		self.parser = argparse.ArgumentParser()
		self.initialized = False
		self.values = {}

	def initialize(self):
		self.parser.add_argument(
			"model", help="one from { baseline-no-aug, baseline, indexed, histogram } - the model to train")
		self.parser.add_argument("--verbose", help="outputs verbosity information",
								 default=False, action="store_true")
		self.parser.add_argument(
			"--source", help="one from { back, left, front, right } - the size used as INPUT", default="front")
		self.parser.add_argument(
			"--target", help="one from { back, left, front, right } - the size used as OUTPUT", default="right")
		self.parser.add_argument("--max-palette-size", type=int,
								 help="the size of the palette to use in the indexed model", default=256)
		self.parser.add_argument("--palette-ordering",
								 help="one from { grayness, top2bottom, bottom2top, shuffled } - the order in which colors appear in an image's palette", default="grayness")
		self.parser.add_argument("--batch", type=int, help="the batch size", default=4)
		self.parser.add_argument(
		    "--lambda_l1", type=float, help="value for lambda_l1 used in baselines and histogram", default=100.)
		self.parser.add_argument("--lambda_segmentation", type=float,
		                         help="value for lambda_segmentation used in indexed mode", default=0.01)
		self.parser.add_argument("--lambda_histogram", type=float,
		                         help="value for lambda_histogram used in histogram mode", default=1.)
		self.parser.add_argument("--epochs", type=int, help="number of epochs to train", default=160)
		self.parser.add_argument("--callback-show-discriminator-output", help="every few update steps, show the discriminator output with some images from the train and test sets", default=False, action="store_true")
		self.parser.add_argument("--callback-evaluate-fid", help="every few update steps, evaluate with the FID metric the performance on the train and test sets", default=False, action="store_true")
		self.parser.add_argument("--callback-evaluate-l1", help="every few update steps, evaluate with the L1 metric the performance on the train and test sets", default=False, action="store_true")
		self.parser.add_argument(
			"--log-folder", help="the folder in which the training procedure saves the logs", default="temp-side2side")
		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.values = self.parser.parse_args()
		setattr(self.values, "source_index", ["back", "left", "front", "right"].index(self.values.source))
		setattr(self.values, "target_index", ["back", "left", "front", "right"].index(self.values.target))

		return self.values

options = OptionParser().parse()

BATCH_SIZE = options.batch
MAX_PALETTE_SIZE = options.max_palette_size
TEMP_FOLDER = options.log_folder

