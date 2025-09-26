from abc import ABC, abstractmethod
from tensorboard.plugins.custom_scalar import layout_pb2, summary as cs_summary
import time
import logging
import gc
import os
import tensorflow as tf

import io_utils
import frechet_inception_distance as fid
from keras_utils import LinearAnnealingScheduler, CosineAnnealingScheduler, ExpCosineAnnealingSchedule, \
    NoopAnnealingScheduler, count_network_parameters
from palette_utils import PaletteExtractor, PaletteExtractorDense, PaletteLossCalculator, PaletteLossCalculatorDense, \
    NoopPaletteLossCalculator, NoopPaletteExtractor


def show_eta(training_start_time, step_start_time, current_step, training_starting_step, total_steps,
             update_steps):
    now = time.time()
    elapsed = now - training_start_time
    steps_so_far = tf.cast(current_step - training_starting_step, tf.float32)
    elapsed_per_step = elapsed / (steps_so_far + 1.)
    remaining_steps = total_steps - steps_so_far
    eta = elapsed_per_step * remaining_steps

    logging.info(f"Time since start: {io_utils.seconds_to_human_readable(elapsed)}")
    logging.info(f"Estimated time to finish: {io_utils.seconds_to_human_readable(eta.numpy())}")
    logging.info(f"Last {update_steps} steps took: {now - step_start_time:.2f}s\n")


class S2SModel(ABC):
    def __init__(self, config):
        self.generator_optimizer = None
        self.discriminator_optimizer = None

        self.best_generator_checkpoint = None
        self.checkpoint_manager = None
        self.summary_writer = None
        self.training_metrics = None

        self.summary_writer = None
        self.now_string = None
        self.log_folders = None

        self.config = config
        self.model_name = config.model_name
        self.experiment = config.experiment
        self.checkpoint_dir = self.get_output_folder("training-checkpoints")
        self.layout_summary = S2SModel.create_layout_summary()

        # initializes palette extractor and loss calculator, which might be no-ops if not using palette quantization
        self.palette_extractor = PaletteExtractor() if config.palette_quantization else NoopPaletteExtractor()
        self.palette_extractor_dense = PaletteExtractorDense() if config.palette_quantization else NoopPaletteExtractor()
        self.palette_loss_calculator = PaletteLossCalculator() if config.palette_quantization else NoopPaletteLossCalculator()
        self.palette_loss_calculator_dense = PaletteLossCalculatorDense() if config.palette_quantization else NoopPaletteLossCalculator()

        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()

        generator_params = count_network_parameters(self.generator)
        discriminator_params = count_network_parameters(self.discriminator)
        logging.info(f"Generator: {self.generator.name} with {generator_params:,} parameters")
        logging.info(f"Discriminator: {self.discriminator.name} with {discriminator_params:,} parameters")

        # if config.palette_quantization, initialize the proper annealing scheduler
        if config.palette_quantization:
            if config.annealing == "linear":
                self.annealing_scheduler = LinearAnnealingScheduler(config.temperature, self.get_annealing_layers())
            elif config.annealing == "cosine":
                # cycles: at least 4 cycles, but can be 60 cycles if steps == 240000
                cycles = max(4, config.steps // 4000)
                self.annealing_scheduler = CosineAnnealingScheduler(config.temperature, cycles,
                                                                    self.get_annealing_layers())
            elif config.annealing == "expcosine":
                cycles = max(4, config.steps // 4000)
                self.annealing_scheduler = ExpCosineAnnealingSchedule(config.temperature, cycles,
                                                                       self.get_annealing_layers())
            else:
                raise ValueError(f"Unknown annealing method: {config.annealing}")
        else:
            self.annealing_scheduler = NoopAnnealingScheduler()

        # initializes training checkpoint information
        io_utils.ensure_folder_structure(self.checkpoint_dir)
        self.best_generator_checkpoint = tf.train.Checkpoint(generator=self.generator)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.best_generator_checkpoint, directory=self.checkpoint_dir, max_to_keep=1)

    def get_output_folder(self, sub_folder=None, skip_run=False, run=None):
        log_folder = self.config.log_folder
        model_name = self.config.model_name
        experiment = self.config.experiment
        run_string = run if not (run is None) else self.config.run_string

        folders = [log_folder, model_name, experiment, run_string]
        if skip_run:
            folders.pop()
        if sub_folder is not None:
            if not isinstance(sub_folder, list):
                sub_folder = [sub_folder]
            folders += sub_folder

        return os.sep.join(folders)

    def save_model_description(self, folder_path):
        io_utils.ensure_folder_structure(folder_path)
        with open(os.sep.join([folder_path, "model-description.txt"]), "w") as fh:
            for model in self.models:
                model.summary(print_fn=lambda x: fh.write(x + "\n"))
                fh.write("\n" * 3)

    @property
    def models(self):
        return [self.discriminator, self.generator]

    def fit(self, train_ds, test_ds, steps, update_steps, callbacks=[], starting_step=0):
        if starting_step == 0:
            # initialize generator and discriminator optimizers
            lr_generator = self.config.lr
            lr_discriminator = self.config.lr
            self.generator_optimizer =  tf.keras.optimizers.Adam(lr_generator, beta_1=0.5)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(lr_discriminator, beta_1=0.5)

            # initializes tensorboard utilities for logging training statistics
            self.summary_writer = tf.summary.create_file_writer(self.get_output_folder())
            with self.summary_writer.as_default():
                tf.summary.experimental.write_raw_pb(
                    self.layout_summary.SerializeToString(), step=0)

            # initialize training metrics (used for saving the best model according to FID or L1)
            self.training_metrics = dict({
                "fid": dict({
                    "best_value": tf.Variable(float("inf"), trainable=False, dtype="float32"),
                    "step": tf.Variable(int(-1), trainable=False, dtype="int64")
                }),
                "l1": dict({
                    "best_value": tf.Variable(float("inf"), trainable=False, dtype="float32"),
                    "step": tf.Variable(int(-1), trainable=False, dtype="int64")
                })
            })

        # does start training
        try:
            self.do_fit(train_ds, test_ds, steps, update_steps, callbacks, starting_step)
        finally:
            self.summary_writer.flush()

    def do_fit(self, train_ds, test_ds, steps, evaluate_steps=1000, callbacks=[], starting_step=0):
        # num_test_images = min(self.config.test_size, 136)
        num_test_images = self.config.test_size
        examples_for_visualization = self.select_examples_for_visualization(train_ds, test_ds)

        example_indices_for_evaluation = []
        examples_for_evaluation = []
        if S2SModel.should_evaluate(callbacks):
            example_indices_for_evaluation = self.initialize_random_examples_for_evaluation(train_ds, test_ds,
                                                                                            num_test_images)

        training_start_time = time.time()
        step_start_time = training_start_time

        for step, batch in train_ds.repeat().take(steps).enumerate():
            step += starting_step

            # every UPDATE_STEPS and in the beginning, visualize x images to see how training is going...
            it_is_time_to_evaluate = (step + 1) % evaluate_steps == 0 or step == 0 or step == steps - 1
            if it_is_time_to_evaluate:
                if step != 0:
                    print("\n")
                    show_eta(training_start_time, step_start_time, step, starting_step, steps, evaluate_steps)

                step_start_time = time.time()

                with self.summary_writer.as_default():
                    save_image_name = os.sep.join([
                        self.get_output_folder(),
                        "step_{:06d},update_{:03d}.png".format(step + 1, (step + 1) // evaluate_steps)
                    ])
                    logging.info(f"Previewing images generated at step {step + 1} (train + test)...")
                    image_data = self.preview_generated_images_during_training(examples_for_visualization,
                                                                               save_image_name, step + 1)
                    image_data = io_utils.plot_to_image(image_data, self.config.output_channels)
                    tf.summary.image(save_image_name, image_data, step=(step + 1) // evaluate_steps, max_outputs=5)

                # check if we need to generate images for evaluation (and do it only once before the callback ifs)
                if S2SModel.should_evaluate(callbacks):
                    logging.info(
                        f"Generating {len(example_indices_for_evaluation['test'][0]) * 2} images for evaluation...")
                    examples_for_evaluation = self.generate_images_for_evaluation(example_indices_for_evaluation)

                # callbacks
                if "debug_discriminator" in callbacks:
                    logging.info("Showing discriminator output patches (3 train + 3 test)...")
                    self.show_discriminated_images(train_ds.unbatch(), "train", step + 1, 3)
                    self.show_discriminated_images(test_ds.unbatch().shuffle(self.config.test_size), "test",
                                                   step + 1, 3)
                if "evaluate_l1" in callbacks:
                    logging.StreamHandler().terminator = ""
                    logging.info(f"Comparing L1 between generated images from train and test...")
                    logging.StreamHandler().terminator = "\n"
                    l1_train, l1_test = self.report_l1(examples_for_evaluation, step=(step + 1) // evaluate_steps)
                    logging.info(f" L1: {l1_train:.5f} / {l1_test:.5f} (train/test)")
                    self.update_training_metrics("l1", l1_test, step + 1, True)

                if "evaluate_fid" in callbacks:
                    logging.info(
                        f"Calculating Fréchet Inception Distance at {(step + 1) / 1000}k with {num_test_images} "
                        f"examples...")
                    fid_train, fid_test = self.report_fid(examples_for_evaluation, step=(step + 1) // evaluate_steps)
                    logging.info(f"FID: {fid_train:.3f} / {fid_test:.3f} (train/test)")
                    self.update_training_metrics("fid", fid_test, step + 1, "evaluate_l1" not in callbacks)

                if S2SModel.should_evaluate(callbacks) and it_is_time_to_evaluate:
                    # free the memory used by the generated examples
                    del examples_for_evaluation
                    examples_for_evaluation = None
                    gc.collect()

                logging.info(f"Step: {(step + 1) / 1000}k")
                if step - starting_step < steps - 1:
                    print("_" * (evaluate_steps // 10))

            # actually TRAIN
            t = tf.cast(step / steps, tf.float32)
            self.train_step(batch, step, evaluate_steps, t)

            # dot feedback for every 10 training steps
            if (step + 1) % 10 == 0 and step - starting_step < steps - 1:
                print(".", end="", flush=True)

        logging.info("\nAbout to exit the training loop...")

        # if no evaluation callback was used, we save a single checkpoint with the end of the training
        if not S2SModel.should_evaluate(callbacks) or self.config.save_last_model:
            self.save_generator_checkpoint(tf.constant(steps, dtype=tf.int32))

    def update_training_metrics(self, metric_name, value, step, should_save_checkpoint=False):
        metric = self.training_metrics[metric_name]
        if value < metric["best_value"]:
            metric["best_value"].assign(value)
            metric["step"].assign(step)
            if should_save_checkpoint:
                self.save_generator_checkpoint(step)

    @staticmethod
    def should_evaluate(callbacks):
        return "evaluate_l1" in callbacks or "evaluate_fid" in callbacks

    @abstractmethod
    def get_annealing_layers(self):
        """
        Returns the layers that will be used for palette quantization annealing.

        Returns:
            A list of layers that will be used for palette quantization annealing. They should have
            a `temperature` attribute
        """
        pass

    @abstractmethod
    def train_step(self, batch, step, update_steps, t):
        pass

    @abstractmethod
    def select_examples_for_visualization(self,  train_ds, test_ds):
        pass

    @abstractmethod
    def preview_generated_images_during_training(self, examples, save_name, step):
        pass

    @abstractmethod
    def initialize_random_examples_for_evaluation(self, train_ds, test_ds, num_images):
        pass

    @abstractmethod
    def generate_images_for_evaluation(self, example_indices_for_evaluation):
        pass

    def extract_palette(self, images):
        """
        Extracts the palette from a batch of images. It assumes the images are in [-1, 1] range.

        Args:
            images (tf.Tensor): A batch of images of shape (batch, height, width, channels) or
            (batch, domains, height, width, channels).

        Returns:
            A tensor of shape (batch, num_colors, channels) containing the extracted palettes,
            or one with a single zero color, in case we're not using palette quantization.

        """
        return self.palette_extractor.extract(images)

    def extract_palette_dense(self, images):
        """
        Extracts the palette from a batch of images. It assumes the images are in [-1, 1] range.

        Args:
            images (tf.Tensor): A batch of images of shape (batch, height, width, channels) or
            (batch, domains, height, width, channels).

        Returns:
            A tensor of shape (batch, num_colors, channels) containing the extracted palettes,
            or one with a single zero color, in case we're not using palette quantization.

        """
        return self.palette_extractor_dense.extract(images)

    def calculate_palette_loss(self, images, palettes, temperature):
        """
        Calculates the palette loss between a batch of images and their corresponding palettes.

        Args:
            images (tf.Tensor): A batch of images of shape (batch, height, width, channels) or
            (batch, domains, height, width, channels).
            palettes (tf.Tensor): A batch of palettes of shape (batch, num_colors, channels).
            temperature (tf.Tensor): A scalar tensor representing the current temperature for annealing.

        Returns:
            A scalar tensor representing the palette loss, or zero if not using palette quantization.
        """
        return self.palette_loss_calculator.calculate(images, palettes, temperature)

    def calculate_palette_loss_dense(self, images, palettes, temperature):
        """
        Calculates the palette loss between a batch of images and their corresponding palettes.

        Args:
            images (tf.Tensor): A batch of images of shape (batch, height, width, channels) or
            (batch, domains, height, width, channels).
            palettes (tf.Tensor): A batch of palettes of shape (batch, num_colors, channels).
            temperature (tf.Tensor): A scalar tensor representing the current temperature for annealing.

        Returns:
            A scalar tensor representing the palette loss, or zero if not using palette quantization.
        """
        return self.palette_loss_calculator_dense.calculate(images, palettes, temperature)

    def evaluate_l1(self, real_image, fake_image):
        return tf.reduce_mean(tf.abs(fake_image - real_image))

    def report_fid(self, examples_for_evaluation, step=None):
        train_real_images, train_fake_images = examples_for_evaluation["train"]
        test_real_images, test_fake_images = examples_for_evaluation["test"]
        train_value = fid.compare(train_real_images.numpy(), train_fake_images.numpy())
        test_value = fid.compare(test_real_images.numpy(), test_fake_images.numpy())

        if hasattr(self, "summary_writer") and step is not None:
            with self.summary_writer.as_default():
                with tf.name_scope("fid"):
                    tf.summary.scalar("train", train_value, step=step,
                                      description=f"Frechét Inception Distance using images "
                                                  f"from the TRAIN dataset")
                    tf.summary.scalar("test", test_value, step=step,
                                      description=f"Frechét Inception Distance using images "
                                                  f"from the TEST dataset")

        return train_value, test_value

    def report_l1(self, examples_for_evaluation, step=None):
        train_real_images, train_fake_images = examples_for_evaluation["train"]
        test_real_images, test_fake_images = examples_for_evaluation["test"]
        train_value = self.evaluate_l1(train_real_images, train_fake_images)
        test_value = self.evaluate_l1(test_real_images, test_fake_images)

        if hasattr(self, "summary_writer") and step is not None:
            with self.summary_writer.as_default():
                with tf.name_scope("l1-evaluation"):
                    tf.summary.scalar("train", train_value, step=step, description=f"L1 between generated and target"
                                                                                   f" images from TRAIN")
                    tf.summary.scalar("test", test_value, step=step, description=f"L1 between generated and target"
                                                                                 f" images from TEST")

        return train_value, test_value

    def restore_best_generator(self):
        self.checkpoint_manager.restore_or_initialize()

        file_path = os.sep.join([self.checkpoint_manager.directory, "step_of_best_generator.txt"])
        if os.path.isfile(file_path):
            file = open(file_path, "r", encoding="utf-8")
            step_of_best_generator = int(next(file))
        else:
            step_of_best_generator = None
        return step_of_best_generator

    def save_generator_checkpoint(self, step):
        self.checkpoint_manager.save()
        file_path = os.sep.join([self.checkpoint_manager.directory, "step_of_best_generator.txt"])
        file = open(file_path, "w", encoding="utf-8")
        file.write(str(step.numpy()))

    def save_generator(self):
        py_model_path = self.get_output_folder(["models", "py", "generator"], True)
        io_utils.delete_folder(py_model_path)
        io_utils.ensure_folder_structure(py_model_path)

        self.generator.save(py_model_path)
        self.save_model_description(py_model_path)

    def load_generator(self):
        py_model_path = self.get_output_folder(["models", "py", "generator"], True)
        self.generator = tf.keras.models.load_model(py_model_path)

    @abstractmethod
    def generate_images_from_dataset(self, dataset, step, num_images=None):
        pass

    @abstractmethod
    def debug_discriminator_output(self, batch, image_path):
        pass

    def show_discriminated_images(self,  dataset, ds_name, step, num_images=3):
        if num_images is None:
            num_images = dataset.cardinality()

        base_path = self.get_output_folder("discriminated-images")
        image_path = os.sep.join([base_path, f"discriminated_{ds_name}_at_step_{step}.png"])
        io_utils.ensure_folder_structure(base_path)

        batch = list(dataset.take(num_images).as_numpy_iterator())
        self.debug_discriminator_output(batch, image_path)

    @staticmethod
    def create_layout_summary():
        return cs_summary.pb(
            layout_pb2.Layout(
                category=[
                    layout_pb2.Category(
                        title="Fréchet Inception Distance",
                        chart=[
                            layout_pb2.Chart(
                                title="FID for train and test",
                                multiline=layout_pb2.MultilineChartContent(
                                    # regex to select only summaries which
                                    # are in "scalar_summaries" name scope:
                                    tag=[r'^fid\/']
                                )
                            )
                        ]
                    ),
                    layout_pb2.Category(
                        title="L1 Evaluation",
                        chart=[
                            layout_pb2.Chart(
                                title="L1 for train and test",
                                multiline=layout_pb2.MultilineChartContent(
                                    # regex to select only summaries which
                                    # are in "scalar_summaries" name scope:
                                    tag=[r'^l1\-evaluation\/']
                                )
                            )
                        ]
                    )
                ]
            )
        )

    @abstractmethod
    def create_discriminator(self):
        pass

    @abstractmethod
    def create_generator(self):
        pass
