import os
import sys
import logging
import tensorflow as tf
from configuration import OptionParser

# instructs matplotlib to use a tmp folder that is outside the network storage on verlab
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

config, parser = OptionParser().parse(sys.argv[1:], True)
logging.info(f"Running with options: {OptionParser.get_description(config, ', ', ':')}")

# configures GPU VRAM usage according to config.vram (limit, default behavior or allow growth on demand)
gpus = tf.config.list_physical_devices("GPU")
requested_gpu = config.gpu
if gpus:
    tf.config.set_visible_devices(gpus[requested_gpu], "GPU")
    if config.vram == -1:
        tf.config.experimental.set_memory_growth(gpus[requested_gpu], True)
    elif config.vram == 0:
        # do nothing -- allow tf to allocate as much as it wants at once
        pass
    else:
        # put a hard limit on the VRAM usage
        tf.config.set_logical_device_configuration(
            gpus[requested_gpu],
            [tf.config.LogicalDeviceConfiguration(memory_limit=config.vram)]
        )

from dataset_utils import load_rgba_ds, load_indexed_ds
from pix2pix_model import Pix2PixModel, Pix2PixAugmentedModel, Pix2PixIndexedModel, Pix2PixHistogramModel
import setup


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
# config.model_name = f"{config.source}-to-{config.target}"
# config.experiment = config.model

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
parser.save_configuration(model.get_output_folder(), sys.argv)

# configuration for training
steps = config.steps
epochs = config.epochs
evaluate_steps = config.evaluate_steps

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

print(f"Saving the generator...")
model.save_generator()

print("Finished executing.")

# python train.py histogram --rm2k --lambda_l1 30 --lambda_histogram 1 --no-aug --histo-loss hellinger --callback-evaluate-fid --callback-evaluate-l1 --batch 1 --log-folder temp-side2side/histogram/histo0,l130,hellinger,b1
# python train.py baseline --rmxp --lambda-l1 100 --no-tran --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --log-folder temp-side2side/postprocess/yuv --post-process yuv
# python train.py baseline --rmxp --lambda-l1 100 --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --log-folder temp-side2side/postprocess/lab,aug --post-process cielab
