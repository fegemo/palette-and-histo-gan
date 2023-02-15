import tensorflow as tf
from math import ceil

from dataset_utils import load_rgba_ds, load_indexed_ds
from configuration import OptionParser
from pix2pix_model import Pix2PixModel, Pix2PixAugmentedModel, Pix2PixIndexedModel, Pix2PixHistogramModel
import setup

options = OptionParser().parse()
if options.verbose:
    print("Running with options: ", options)
    print("Tensorflow version: ", tf.__version__)

    if tf.test.gpu_device_name():
        print("Default GPU: {}".format(tf.test.gpu_device_name()))
    else:
        print("Not using a GPU - it will take long!!")

# check if datasets need unzipping
if options.verbose:
    print("Datasets used: ", options.datasets_used)
setup.ensure_datasets(options.verbose)

# setting the seed
if options.verbose:
    print("SEED set to: ", options.seed)
tf.random.set_seed(options.seed)

# loading the dataset according to the required model
if options.model == "baseline-no-aug":
    train_ds, test_ds = load_rgba_ds(
        options.source_index, options.target_index, augment=False)
elif options.model == "baseline":
    train_ds, test_ds = load_rgba_ds(options.source_index, options.target_index, augment=(not options.no_aug))
elif options.model == "indexed":
    train_ds, test_ds = load_indexed_ds(
        options.source_index, options.target_index, palette_ordering=options.palette_ordering)
elif options.model == "histogram":
    train_ds, test_ds = load_rgba_ds(options.source_index, options.target_index, augment=(not options.no_aug))
else:
    raise SystemExit(
        f"The specified model {options.model} was not recognized.")

# instantiates the proper model
architecture_name = f"{options.source}-to-{options.target}"

if options.model == "baseline-no-aug":
    model = Pix2PixModel(
        train_ds=train_ds,
        test_ds=test_ds,
        model_name="baseline (no aug.)",
        architecture_name=architecture_name,
        lambda_l1=options.lambda_l1,
        keep_checkpoint=options.keep_checkpoint)
elif options.model == "baseline":
    model = Pix2PixAugmentedModel(
        train_ds=train_ds,
        test_ds=test_ds,
        model_name="baseline",
        architecture_name=architecture_name,
        lambda_l1=options.lambda_l1,
        keep_checkpoint=options.keep_checkpoint)
elif options.model == "indexed":
    model = Pix2PixIndexedModel(
        train_ds=train_ds,
        test_ds=test_ds,
        model_name="indexed",
        architecture_name=architecture_name,
        lambda_segmentation=options.lambda_segmentation,
        keep_checkpoint=options.keep_checkpoint)
elif options.model == "histogram":
    model = Pix2PixHistogramModel(
        train_ds=train_ds,
        test_ds=test_ds,
        model_name="histogram",
        architecture_name=architecture_name,
        lambda_l1=options.lambda_l1,
        lambda_histogram=options.lambda_histogram,
        keep_checkpoint=options.keep_checkpoint)

# configuration for training
steps = ceil(train_ds.cardinality() / options.batch) * options.epochs
update_steps = steps // 40

print(
    f"Starting training for {options.epochs} epochs in {steps} steps, updating visualization every "
    f"{update_steps} steps...")

# starting training
callbacks = [c[len("callback_"):] for c in ["callback_show_discriminator_output", "callback_evaluate_fid",
                                            "callback_evaluate_l1"] if
             getattr(options, c)]

model.fit(steps, update_steps, callbacks=callbacks)

# generating resulting images
model.generate_images_from_dataset()

if options.save_model:
    print(f"Saving the generator...")
    model.save_generator()

print("Finished executing.")
