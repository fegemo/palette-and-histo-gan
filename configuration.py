import argparse
import os
import sys
import datetime
from math import ceil

SEED = 47

DATASET_NAMES = ["tiny-hero", "rpg-maker-2000", "rpg-maker-xp", "rpg-maker-vxace", "miscellaneous"]
DATA_FOLDERS = [
    os.sep.join(["datasets", folder])
    for folder
    in DATASET_NAMES
]

DIRECTIONS = ["back", "left", "front", "right"]
DIRECTION_BACK = DIRECTIONS.index("back")  # 0
DIRECTION_LEFT = DIRECTIONS.index("left")  # 1
DIRECTION_FRONT = DIRECTIONS.index("front")  # 2
DIRECTION_RIGHT = DIRECTIONS.index("right")  # 3
# ["0-back", "1-left", "2-front", "3-right"]
DIRECTION_FOLDERS = [f"{i}-{name}" for i, name in enumerate(DIRECTIONS)]
TRAIN_PERCENTAGE = 0.85


BATCH_SIZE = 4
IMG_SIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

# for indexed colors
MAX_PALETTE_SIZE = 256
INVALID_INDEX_COLOR = [255, 0, 220, 255]  # some hotpink

LOG_FOLDER = "temp-side2side"


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
        self.parser.add_argument("--image-size", help="size of squared images", default=IMG_SIZE, type=int)
        self.parser.add_argument("--output-channels", help="size of squared images", default=OUTPUT_CHANNELS, type=int)
        self.parser.add_argument("--input-channels", help="size of squared images", default=INPUT_CHANNELS, type=int)
        self.parser.add_argument("--verbose", help="outputs verbosity information",
                                 default=False, action="store_true")

        self.parser.add_argument("--rmxp", action="store_true", default=False, help="Uses RPG Maker XP dataset")
        self.parser.add_argument("--rm2k", action="store_true", default=False, help="Uses RPG Maker 2000"
                                                                                    " dataset")
        self.parser.add_argument("--rmvx", action="store_true", default=False, help="Uses RPG Maker VX Ace"
                                                                                    " dataset")
        self.parser.add_argument("--tiny", action="store_true", default=False, help="Uses the Tiny Hero dataset")
        self.parser.add_argument("--misc", action="store_true", default=False, help="Uses the miscellaneous"
                                                                                    " sprites dataset")
        self.parser.add_argument("--rmxp-validation", action="store_true", default=False, help="Uses only RMXP (44 "
                                                                                                "test examples) for "
                                                                                                "validation and to "
                                                                                                "generate images in "
                                                                                                "the end")
        self.parser.add_argument("--rm2k-validation", action="store_true", default=False, help="Uses only RM2K (32 "
                                                                                                "test examples) for "
                                                                                                "validation and to "
                                                                                                "generate images in "
                                                                                                "the end")
        self.parser.add_argument("--rmvx-validation", action="store_true", default=False, help="Uses only RMVX (61 "
                                                                                                "test examples) for "
                                                                                                "validation and to "
                                                                                                "generate images in "
                                                                                                "the end")
        self.parser.add_argument("--tiny-validation", action="store_true", default=False, help="Uses only tiny (136 "
                                                                                                "test examples) for "
                                                                                                "validation and to "
                                                                                                "generate images in "
                                                                                                "the end")
        self.parser.add_argument(
            "--source", help="one from { back, left, front, right } - the size used as INPUT", default="front")
        self.parser.add_argument(
            "--target", help="one from { back, left, front, right } - the size used as OUTPUT", default="right")

        self.parser.add_argument("--max-palette-size", type=int,
                                 help="the size of the palette to use in the indexed model", default=256)
        self.parser.add_argument("--palette-ordering",
                                 help="one from { grayness, top2bottom, bottom2top, shuffled } - the order in which "
                                      "colors appear in an image's palette",
                                 default="grayness")
        self.parser.add_argument("--batch", type=int, help="the batch size", default=4)
        self.parser.add_argument(
            "--lambda-l1", type=float, help="value for lambda_l1 used in baselines and histogram", default=100.)
        self.parser.add_argument("--lambda-segmentation", type=float,
                                 help="value for lambda-segmentation used in indexed mode", default=0.01)
        self.parser.add_argument("--lambda-histogram", type=float,
                                 help="value for lambda-histogram used in histogram mode", default=1.)
        self.parser.add_argument("--lr", type=float, help="learning rate", default=0.0002)
        self.parser.add_argument("--epochs", type=int, help="number of epochs to train", default=160)
        self.parser.add_argument("--no-aug", action="store_true", help="Disables all augmentation", default=False)
        self.parser.add_argument("--no-hue", action="store_true", help="Disables hue augmentation", default=False)
        self.parser.add_argument("--no-tran", action="store_true", help="Disables translation augmentation",
                                 default=False)
        self.parser.add_argument("--histo-loss", help="one of { hellinger, l1, l2 } to use as histogram loss",
                                 default="hellinger")
        self.parser.add_argument("--callback-show-discriminator-output",
                                 help="every few update steps, show the discriminator output with some images from "
                                      "the train and test sets",
                                 default=False, action="store_true")
        self.parser.add_argument("--callback-evaluate-fid",
                                 help="every few update steps, evaluate with the FID metric the performance "
                                      "on the train and test sets",
                                 default=False, action="store_true")
        self.parser.add_argument("--callback-evaluate-l1",
                                 help="every few update steps, evaluate with the L1 metric the performance "
                                      "on the train and test sets",
                                 default=False, action="store_true")
        self.parser.add_argument("--save-last-model", help="saves the model at the end instead of the best one", default=False, action="store_true")
        self.parser.add_argument("--model-name", help="architecture name", default="some-architecture")
        self.parser.add_argument("--experiment", help="description of this experiment", default="playground")
        self.parser.add_argument(
            "--log-folder", help="the folder in which the training procedure saves the logs", default="temp-side2side")
        self.parser.add_argument("--post-process", help="post-processes the generated images using one from { none, rgb, yuv, cielab }", default="none")
        self.initialized = True

    def parse(self, args=None, return_parser=False):
        if args is None:
            args = sys.argv[1:]
        if not self.initialized:
            self.initialize()
        self.values = self.parser.parse_args(args)

        setattr(self.values, "source_index", ["back", "left", "front", "right"].index(self.values.source))
        setattr(self.values, "target_index", ["back", "left", "front", "right"].index(self.values.target))
        setattr(self.values, "seed", SEED)
        if self.values.no_aug:
            setattr(self.values, "no_hue", True)
            setattr(self.values, "no_tran", True)
        datasets_used = list(filter(lambda opt: getattr(self.values, opt), ["tiny", "rm2k", "rmxp", "rmvx", "misc"]))
        setattr(self.values, "datasets_used", datasets_used)
        if len(datasets_used) == 0:
            raise Exception("No dataset was supplied with: --tiny, --rm2k, --rmxp, --rmvx, --misc")
        setattr(self.values, "dataset_names", DATASET_NAMES)
        setattr(self.values, "data_folders", [
            os.sep.join(["datasets", folder])
            for folder
            in self.values.dataset_names
        ])
        setattr(self.values, "run_string", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        dataset_mask = list(
            map(lambda opt: 1 if getattr(self.values, opt) else 0, ["tiny", "rm2k", "rmxp", "rmvx", "misc"]))
        dataset_sizes = [912, 216, 294, 408, 12372]
        dataset_sizes = [n * m for n, m in zip(dataset_sizes, dataset_mask)]
        train_sizes = [ceil(n * TRAIN_PERCENTAGE) for n in dataset_sizes]
        train_size = sum(train_sizes)
        test_sizes = [dataset_sizes[i] - train_sizes[i]
                      for i, n in enumerate(dataset_sizes)]
        if self.values.rmxp_validation:
            test_sizes = [0, 0, 44, 0, 0]
        elif self.values.rm2k_validation:
            test_sizes = [0, 32, 0, 0, 0]
        elif self.values.rmvx_validation:
            test_sizes = [0, 0, 0, 61, 0]
        elif self.values.tiny_validation:
            test_sizes = [136, 0, 0, 0, 0]

        test_size = sum(test_sizes)

        setattr(self.values, "dataset_sizes", dataset_sizes)
        setattr(self.values, "dataset_mask", dataset_mask)
        setattr(self.values, "train_sizes", train_sizes)
        setattr(self.values, "train_size", train_size)
        setattr(self.values, "test_sizes", test_sizes)
        setattr(self.values, "test_size", test_size)


        if return_parser:
            return self.values, self
        else:
            return self.values

    def get_description(self, param_separator=",", key_value_separator="-"):
        sorted_args = sorted(vars(self.values).items())
        description = param_separator.join(map(lambda p: f"{p[0]}{key_value_separator}{p[1]}", sorted_args))
        return description

    def save_configuration(self, folder_path):
        from io_utils import ensure_folder_structure
        ensure_folder_structure(folder_path)
        with open(os.sep.join([folder_path, "configuration.txt"]), "w") as file:
            file.write(self.get_description("\n", ": ") + "\n")


def in_notebook():
    try:
        from IPython import get_ipython
        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


# if not in_notebook():
#     options = OptionParser().parse(sys.argv)
#
#     BATCH_SIZE = options.batch
#     MAX_PALETTE_SIZE = options.max_palette_size
#     LOG_FOLDER = options.log_folder
