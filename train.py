import sys

import tensorflow as tf
from math import ceil

from dataset_utils import load_rgba_ds, load_indexed_ds
from configuration import OptionParser
from pix2pix_model import Pix2PixModel, Pix2PixAugmentedModel, Pix2PixIndexedModel, Pix2PixHistogramModel
import setup

config, parser = OptionParser().parse(sys.argv[1:], True)
if config.verbose:
    print("Running with options: ", config)
    print("Tensorflow version: ", tf.__version__)

    if tf.test.gpu_device_name():
        print("Default GPU: {}".format(tf.test.gpu_device_name()))
    else:
        print("Not using a GPU - it will take long!!")

# check if datasets need unzipping
if config.verbose:
    print("Datasets used: ", config.datasets_used)
setup.ensure_datasets(config.verbose)

# setting the seed
if config.verbose:
    print("SEED set to: ", config.seed)
tf.random.set_seed(config.seed)

# loading the dataset according to the required model
if config.model in ["baseline-no-aug", "baseline", "histogram"]:
    train_ds, test_ds = load_rgba_ds(config)
elif config.model == "indexed":
    train_ds, test_ds = load_indexed_ds(config)
else:
    raise SystemExit(
        f"The specified model {config.model} was not recognized.")

# instantiates the proper model
config.model_name = f"{config.source}-to-{config.target}"
config.experiment = config.model

if config.model == "baseline-no-aug":
    class_name = Pix2PixModel
elif config.model == "baseline":
    class_name = Pix2PixAugmentedModel
elif config.model == "indexed":
    class_name = Pix2PixIndexedModel
elif config.model == "histogram":
    class_name = Pix2PixHistogramModel
else:
    raise Exception(f"The asked model of {config.model} was not found.")

model = class_name(config)

model.save_model_description(model.get_output_folder())
if config.verbose:
    model.discriminator.summary()
    model.generator.summary()
parser.save_configuration(model.get_output_folder())

# configuration for training
steps = ceil(config.train_size / config.batch) * config.epochs
evaluate_steps = steps // 40
# evaluate_steps = 500

print(
    f"Starting training for {config.epochs} epochs in {steps} steps, updating visualization every "
    f"{evaluate_steps} steps...")

# starting training
callbacks = [c[len("callback_"):] for c in ["callback_show_discriminator_output", "callback_evaluate_fid",
                                            "callback_evaluate_l1"] if
             getattr(config, c)]

model.fit(train_ds, test_ds, steps, evaluate_steps, callbacks=callbacks)

# restores the best generator (best l1 - priority, or best fid)
step = model.restore_best_generator()
print(f"Restored the BEST generator, which was in step {step}.")

# generating resulting images
print(f"Starting to generate the images from the test dataset with generator from step {step}...")
model.generate_images_from_dataset(test_ds, step)

if config.save_model:
    print(f"Saving the generator...")
    model.save_generator()

print("Finished executing.")

# python train.py histogram --rm2k --lambda_l1 30 --lambda_histogram 1 --no-aug --histo-loss hellinger --callback-evaluate-fid --callback-evaluate-l1 --batch 1 --log-folder temp-side2side/histogram/histo0,l130,hellinger,b1
# python train.py baseline --rmxp --lambda-l1 100 --no-tran --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --log-folder temp-side2side/postprocess/yuv --post-process yuv
# python train.py baseline --rmxp --lambda-l1 100 --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --log-folder temp-side2side/postprocess/lab,aug --post-process cielab
