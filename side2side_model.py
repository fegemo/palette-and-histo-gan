from abc import ABC, abstractmethod
import tensorflow as tf
from tensorboard.plugins.custom_scalar import layout_pb2, summary as cs_summary
import time
import datetime
from IPython import display
from matplotlib import pyplot as plt

import io_utils
from configuration import *
import frechet_inception_distance as fid


def show_eta(training_start_time, step_start_time, current_step, training_starting_step, total_steps,
             update_steps):
    now = time.time()
    elapsed = now - training_start_time
    steps_so_far = tf.cast(current_step - training_starting_step, tf.float32)
    elapsed_per_step = elapsed / (steps_so_far + 1.)
    remaining_steps = total_steps - steps_so_far
    eta = elapsed_per_step * remaining_steps

    print(f"Time since start: {io_utils.seconds_to_human_readable(elapsed)}")
    print(f"Estimated time to finish: {io_utils.seconds_to_human_readable(eta.numpy())}")
    print(f"Last {update_steps} steps took: {now - step_start_time:.2f}s\n")


class S2SModel(ABC):
    def __init__(self, train_ds, test_ds, model_name, architecture_name="s2smodel"):
        """
        Params:
        - train_ds: the dataset used for training. Should have target images as labels
        - test_ds: dataset used for validation. Should have target images as labels
        - model_name: the specific direction of source to target image (eg, front2right).
                      Should be path-friendly
        - architecture_name: the network architecture + variation used (eg, pix2pix, pix2pix-wgan).
                             Should be path-friendly
        """
        self.generator = None
        self.discriminator = None

        self.summary_writer = None
        self.now_string = None
        self.log_folders = None

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.model_name = model_name
        self.architecture_name = architecture_name
        self.checkpoint_dir = os.sep.join(
            [TEMP_FOLDER, "training-checkpoints", self.architecture_name, self.model_name])
        self.layout_summary = S2SModel.create_layout_summary()

    def fit(self, steps, update_steps, callbacks=[], starting_step=0):
        if starting_step == 0:
            self.log_folders = [TEMP_FOLDER, "logs", self.architecture_name, self.model_name]
            self.now_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.summary_writer = tf.summary.create_file_writer(os.sep.join([*self.log_folders, self.now_string]))
            with self.summary_writer.as_default():
                tf.summary.experimental.write_raw_pb(
                    self.layout_summary.SerializeToString(), step=0)
        try:
            self.do_fit(steps, update_steps, callbacks, starting_step)
        finally:
            self.summary_writer.flush()

    def do_fit(self, steps, update_steps=1000, callbacks=[], starting_step=0):
        examples = self.select_examples_for_visualization()

        training_start_time = time.time()
        step_start_time = training_start_time

        for step, batch in self.train_ds.repeat().take(steps).enumerate():
            step += starting_step

            # every UPDATE_STEPS and in the beginning, visualize 5x images to see
            # how training is going...
            if (step + 1) % update_steps == 0 or step == 0:
                display.clear_output(wait=True)

                if step != 0:
                    show_eta(training_start_time, step_start_time, step, starting_step, steps, update_steps)

                step_start_time = time.time()

                with self.summary_writer.as_default():
                    save_image_name = os.sep.join(
                        [TEMP_FOLDER, "logs", self.architecture_name, self.model_name, self.now_string,
                         "step_{:06d}.png".format(step + 1)])
                    print(f"Previewing images generated at step {step + 1} (3 test + 3 train)...")
                    image_data = self.preview_generated_images_during_training(examples, save_image_name, step + 1)
                    image_data = io_utils.plot_to_image(image_data)
                    tf.summary.image(save_image_name, image_data, step=(step + 1) // update_steps, max_outputs=5)

                if "show_discriminator_output" in callbacks:
                    print("Showing discriminator output patches (2 test + 2 train)...")
                    self.show_discriminated_images("test", 2)
                    self.show_discriminated_images("train", 2)
                if "evaluate_l1" in callbacks:
                    print(f"Comparing L1 between generated images from train and test...", end="", flush=True)
                    l1_train, l1_test = self.report_l1(step=(step + 1) // update_steps)
                    print(f" L1: {l1_train:.5f} / {l1_test:.5f} (train/test)")
                if "evaluate_fid" in callbacks:
                    print(
                        f"Calculating Fréchet Inception Distance at {(step + 1) / 1000}k with {TEST_SIZE} examples...")
                    train_fid, test_fid = self.report_fid(step=(step + 1) // update_steps)
                    print(f"FID: {train_fid:.3f} / {test_fid:.3f} (train/test)")

                print(f"Step: {(step + 1) / 1000}k")
                if step - starting_step < steps - 1:
                    print("˯" * (update_steps // 10))

            # actually TRAIN
            self.train_step(batch, step, update_steps)

            # dot feedback for every 10 training steps
            if (step + 1) % 10 == 0 and step - starting_step < steps - 1:
                print(".", end="", flush=True)

            # saves a training checkpoint UPDATE_STEPS*5
            if (step + 1) % (update_steps * 5) == 0 or (step - starting_step + 1) == steps:
                self.checkpoint_manager.save()

    @abstractmethod
    def train_step(self, batch, step, UPDATE_STEPS):
        pass

    @abstractmethod
    def select_examples_for_visualization(self, number_of_examples=6):
        pass

    @abstractmethod
    def preview_generated_images_during_training(self, examples, save_name, step):
        pass

    @abstractmethod
    def select_examples_for_evaluation(self, num_images, dataset):
        pass

    @abstractmethod
    def evaluate_l1(self, real_images, fake_images):
        pass

    def report_fid(self, num_images=TEST_SIZE, step=None):
        train_real_images, train_fake_images = self.select_examples_for_evaluation(num_images, self.train_ds)
        test_real_images, test_fake_images = self.select_examples_for_evaluation(num_images, self.test_ds)
        train_value = fid.compare(train_real_images, train_fake_images)
        test_value = fid.compare(test_real_images, test_fake_images)

        if hasattr(self, "summary_writer") and step is not None:
            with self.summary_writer.as_default():
                with tf.name_scope("fid"):
                    tf.summary.scalar("train", train_value, step=step,
                                      description=f"Frechét Inception Distance using {num_images} images "
                                                  f"from the TRAIN dataset")
                    tf.summary.scalar("test", test_value, step=step,
                                      description=f"Frechét Inception Distance using {num_images} images "
                                                  f"from the TEST dataset")

        return train_value, test_value

    def report_l1(self, num_images=TEST_SIZE, step=None):
        train_real_images, train_fake_images = self.select_examples_for_evaluation(num_images, self.train_ds)
        test_real_images, test_fake_images = self.select_examples_for_evaluation(num_images, self.test_ds)
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

    def save_generator(self):
        py_model_path = os.sep.join([TEMP_FOLDER, "models", "py", "generator", self.architecture_name, self.model_name])

        io_utils.delete_folder(py_model_path)
        io_utils.ensure_folder_structure(py_model_path)

        self.generator.save(py_model_path)

    def load_generator(self):
        self.generator = tf.keras.models.load_model(
            os.sep.join(["models", "py", "generator", self.architecture_name, self.model_name]))

    def save_discriminator(self):
        py_model_path = os.sep.join(["models", "py", "discriminator", self.architecture_name, self.model_name])

        io_utils.delete_folder(py_model_path)
        io_utils.ensure_folder_structure(py_model_path)

        self.discriminator.save(py_model_path)

    def load_discriminator(self):
        self.discriminator = tf.keras.models.load_model(
            os.sep.join(["models", "py", "discriminator", self.architecture_name, self.model_name]))

    def generate_images_from_dataset(self, dataset_name="test", num_images=None, steps=None):
        is_test = dataset_name == "test"

        if num_images is None:
            num_images = TEST_SIZE if is_test else TRAIN_SIZE
        num_images = min(num_images, TEST_SIZE if is_test else TRAIN_SIZE)

        dataset = self.test_ds if is_test else self.train_ds
        dataset = list(dataset.unbatch().take(num_images).batch(1).as_numpy_iterator())

        base_image_path = os.sep.join([TEMP_FOLDER, "generated-images", self.architecture_name, self.model_name])

        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)

        for i, images in enumerate(dataset):
            image_path = os.sep.join([base_image_path, f"{i}.png"])
            fig = self.preview_generated_images_during_training([images], image_path, steps)
            plt.close(fig)

        print(f"Generated {i + 1} images (using \"{dataset_name}\" dataset)")

    @abstractmethod
    def debug_discriminator_patches(self, batch_of_one):
        pass

    def show_discriminated_images(self, dataset_name="test", num_images=2):
        is_test = dataset_name == "test"

        if num_images is None:
            num_images = min(num_images, TEST_SIZE if is_test else TRAIN_SIZE)

        dataset = self.test_ds if is_test else self.train_ds
        dataset = list(dataset.unbatch().take(num_images).batch(1).as_numpy_iterator())

        for images in dataset:
            self.debug_discriminator_patches(images)

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
