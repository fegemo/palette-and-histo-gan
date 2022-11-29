import tensorflow as tf
from math import ceil

from dataset_utils import load_rgba_ds, load_indexed_ds
from configuration import OptionParser
from pix2pix_model import Pix2PixModel, Pix2PixAugmentedModel, Pix2PixIndexedModel, Pix2PixHistogramModel

options = OptionParser().parse()
if options.verbose:
	print("Running with options: ", options)
	print("Tensorflow version: ", tf.__version__)

	if tf.test.gpu_device_name():
		print("Default GPU: {}".format(tf.test.gpu_device_name()))
	else:
		print("Not using a GPU - it will take long!!")


# loading the dataset according to the required model
if options.model == "baseline-no-aug":
    train_ds, test_ds = load_rgba_ds(
        options.source_index, options.target_index, augment=False)
elif options.model == "baseline":
	train_ds, test_ds = load_rgba_ds(options.source_index, options.target_index)
elif options.model == "indexed":
	train_ds, test_ds = load_indexed_ds(
		options.source_index, options.target_index, palette_ordering=options.palette_ordering)
elif options.model == "histogram":
	train_ds, test_ds = load_rgba_ds(options.source_index, options.target_index)
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
		lambda_l1=options.lambda_l1)
elif options.model == "baseline":
	model = Pix2PixAugmentedModel(
		train_ds=train_ds,
		test_ds=test_ds,
		model_name="baseline",
		architecture_name=architecture_name,
		lambda_l1=options.lambda_l1)
elif options.model == "indexed":
	model = Pix2PixIndexedModel(
		train_ds=train_ds,
		test_ds=test_ds,
		model_name="indexed",
		architecture_name=architecture_name,
		lambda_segmentation=options.lambda_segmentation)
elif options.model == "histogram":
	model = Pix2PixHistogramModel(
		train_ds=train_ds,
		test_ds=test_ds,
		model_name="histogram",
		architecture_name=architecture_name,
		lambda_l1=options.lambda_l1,
		lambda_histogram=options.lambda_histogram)


# configuration for training
DATASET_SIZES = [294]
DATASET_SIZE = sum(DATASET_SIZES)
TRAIN_PERCENTAGE = 0.85
TRAIN_SIZES = [ceil(n * TRAIN_PERCENTAGE) for n in DATASET_SIZES]
TRAIN_SIZE = sum(TRAIN_SIZES)
TEST_SIZES = [DATASET_SIZES[i] - TRAIN_SIZES[i]
              for i, n in enumerate(DATASET_SIZES)]
TEST_SIZE = sum(TEST_SIZES)

steps = ceil(TRAIN_SIZE / options.batch) * options.epochs
update_steps = steps // 40

print(
	f"Starting training for {options.epochs} epochs in {steps} steps, updating visualization every {update_steps} steps...")


# starting training
callbacks = [c.replace("-", "_")[len("callback_"):] for c in ["callback-show-discriminator-output",
                                                              "callback-evaluate-fid", "callback-evaluate-l1"] if c in options]

model.fit(steps, update_steps, callbacks=callbacks)


# generating resulting images
model.generate_images_from_dataset()
