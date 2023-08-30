import numpy as np
from IPython import display
from matplotlib import pyplot as plt
import tensorflow_io as tfio
from scipy.spatial import KDTree

import histogram
import io_utils
from dataset_utils import blacken_transparent_pixels
from networks import *
from side2side_model import S2SModel


class PostProcessGenerator(tf.keras.Model):
    def __init__(self, real_generator, post_process_type):
        super().__init__()
        self.real_generator = real_generator
        self.post_process_type = post_process_type

    def __call__(self, batch, **kwargs):
        fake_image = self.real_generator(batch, **kwargs)
        palette = io_utils.batch_extract_palette(batch)
        post_processed_fake_image = self.quantize_to_palette(fake_image, palette)

        return post_processed_fake_image

    def quantize_to_palette(self, batch_image, batch_palette):
        # batch_image and batch_palette come in [-1, 1]
        batch_palette_original = batch_palette
        # but they must be in [0, 1] for conversion to lab/yuv
        batch_image = batch_image * 0.5 + 0.5
        batch_palette = batch_palette * 0.5 + 0.5

        batch_image_rgb = batch_image[..., :3]
        batch_image_alpha = batch_image[..., 3:]
        batch_palette_rgb = batch_palette[..., :3]
        batch_palette_alpha = batch_palette[..., 3:]
        if self.post_process_type == "cielab":
            batch_image_lab = tfio.experimental.color.rgb_to_lab(batch_image_rgb)
            batch_image = tf.concat([batch_image_lab, batch_image_alpha], -1)
            batch_palette_lab = tfio.experimental.color.rgb_to_lab(batch_palette_rgb)
            batch_palette = tf.concat([batch_palette_lab, batch_palette_alpha], -1)
        elif self.post_process_type == "yuv":
            batch_image_yuv = tfio.experimental.color.rgb_to_yuv(batch_image[..., :3])
            batch_image = tf.concat([batch_image_yuv, batch_image_alpha], -1)
            batch_palette_yuv = tfio.experimental.color.rgb_to_yuv(batch_palette_rgb)
            batch_palette = tf.concat([batch_palette_yuv, batch_palette_alpha], -1)

        batch_image = batch_image.numpy()
        batch_palette = batch_palette.numpy()
        batch_palette_original = batch_palette_original.numpy()

        results = []
        for image, palette, palette_original in zip(batch_image, batch_palette, batch_palette_original):
            # creates a tree of similar colors
            palette_tree = KDTree(palette)
            # finds the closest color index for each pixel
            _, indices = palette_tree.query(image)
            # creates the image quantized to the palette
            result = palette_original[indices]
            # adds the just palette-quantized image to the results batch
            results.append(result)

        results = tf.stack(results)
        return results


class Pix2PixModel(S2SModel):
    def __init__(self, config):
        super().__init__(config)

        self.lambda_l1 = config.lambda_l1
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def create_generator(self):
        real_generator = UnetGenerator(4, 4, "tanh")
        if self.config.post_process is not None and self.config.post_process != "none":
            self.proxy_generator = PostProcessGenerator(real_generator, self.config.post_process)
        else:
            self.proxy_generator = real_generator
        return real_generator

    def create_discriminator(self):
        return PatchDiscriminator(4)

    def generator_loss(self, fake_predicted, fake_image, real_image):
        adversarial_loss = self.loss_object(tf.ones_like(fake_predicted), fake_predicted)
        l1_loss = tf.reduce_mean(tf.abs(real_image - fake_image))
        total_loss = adversarial_loss + (self.lambda_l1 * l1_loss)

        return total_loss, adversarial_loss, l1_loss

    def discriminator_loss(self, real_predicted, fake_predicted):
        real_loss = self.loss_object(tf.ones_like(real_predicted), real_predicted)
        fake_loss = self.loss_object(tf.zeros_like(fake_predicted), fake_predicted)
        total_loss = fake_loss + real_loss

        return total_loss, real_loss, fake_loss

    def generate(self, batch):
        source_image, _ = batch
        return self.proxy_generator(source_image, training=True)

    @tf.function
    def train_step(self, batch, step, evaluate_steps, t):
        source_image, real_image = batch

        with tf.GradientTape(persistent=True) as tape:
            fake_image = self.generator(source_image, training=True)

            real_predicted = self.discriminator([real_image, source_image], training=True)
            fake_predicted = self.discriminator([fake_image, source_image], training=True)

            g_loss = self.generator_loss(fake_predicted, fake_image, real_image)
            generator_total_loss = g_loss[0]

            d_loss = self.discriminator_loss(real_predicted, fake_predicted)
            discriminator_total_loss = d_loss[0]

        generator_gradients = tape.gradient(generator_total_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(discriminator_total_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            with tf.name_scope("generator"):
                self.log_generator_loss(g_loss, step // evaluate_steps)
            with tf.name_scope("discriminator"):
                self.log_discriminator_loss(d_loss, step // evaluate_steps)

    def log_generator_loss(self, g_loss, step):
        total_loss, adversarial_loss, l1_loss = g_loss
        tf.summary.scalar("total_loss", total_loss, step=step)
        tf.summary.scalar("adversarial_loss", adversarial_loss, step=step)
        tf.summary.scalar("l1_loss", l1_loss, step=step)

    def log_discriminator_loss(self, d_loss, step):
        total_loss, real_loss, fake_loss = d_loss
        tf.summary.scalar("total_loss", total_loss, step=step)
        tf.summary.scalar("real_loss", real_loss, step=step)
        tf.summary.scalar("fake_loss", fake_loss, step=step)

    def select_examples_for_visualization(self, train_ds, test_ds):
        num_train_examples = 3
        num_test_examples = 3

        train_examples = train_ds.unbatch().take(num_train_examples).batch(1)
        test_examples = test_ds.unbatch().take(num_test_examples).batch(1)

        return list(train_examples.as_numpy_iterator()) + list(test_examples.as_numpy_iterator())

    def select_examples_for_evaluation(self, num_images, dataset):
        real_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, 4))
        fake_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, 4))
        dataset = dataset.unbatch().take(num_images).batch(1)

        for i, (source_image, real_image) in dataset.enumerate():
            fake_image = self.proxy_generator(source_image, training=True)
            real_images[i] = real_image[0].numpy()
            fake_images[i] = fake_image[0].numpy()

        return real_images, fake_images

    def initialize_random_examples_for_evaluation(self, train_ds, test_ds, num_images):
        def initialize_random_examples_from_dataset(dataset):
            source_images, target_images = next(iter(dataset.unbatch().batch(num_images).take(1)))
            return target_images, source_images

        return dict({
            "train": initialize_random_examples_from_dataset(train_ds),
            "test": initialize_random_examples_from_dataset(test_ds.shuffle(self.config.test_size))
        })

    def generate_images_for_evaluation(self, example_indices_for_evaluation):
        generator = self.generator
        def generate_images_from_dataset(dataset_name):
            target_images, source_images = example_indices_for_evaluation[dataset_name]
            fake_images = generator(source_images, training=True)
            return target_images, fake_images

        return dict({
            "train": generate_images_from_dataset("train"),
            "test": generate_images_from_dataset("test")
        })

    def evaluate_l1(self, real_images, fake_images):
        return tf.reduce_mean(tf.abs(fake_images - real_images))

    def preview_generated_images_during_training(self, examples, save_name, step):
        has_postprocess_columns = self.config.post_process != "none"
        title = ["Input", "Target", "Generated", "Input histo", "Target histo", "Generated histo"]
        if has_postprocess_columns:
            title = title[:3] + ["Post-processed"] + title[3:] + ["Pstpcssd histo"]
        num_images = len(examples)
        num_columns = len(title)

        if step is not None:
            title[-1] += f" ({step / 1000}k)"
        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))

        predicted_images = []
        source_image_histograms = [histogram.calculate_rgbuv_histogram(image[0]) for image in examples]
        target_image_histograms = [histogram.calculate_rgbuv_histogram(image[1]) for image in examples]
        predicted_images_histograms = []
        post_processed_images_histograms = []

        for i, (source_image, target_image) in enumerate(examples):
            if i >= len(predicted_images):
                predicted_image = self.generator(source_image, training=True)
                predicted_images.append(predicted_image)
                predicted_images_histograms.append(
                    histogram.calculate_rgbuv_histogram(predicted_image))

            images = [source_image, target_image, predicted_images[i]]
            if has_postprocess_columns:
                post_processed_image = self.proxy_generator(source_image, training=True)
                images += [post_processed_image]
                post_processed_images_histograms.append(
                    histogram.calculate_rgbuv_histogram(post_processed_image))

            for j in range(len(images)):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                plt.title(title[j] if i == 0 else "", fontdict={"fontsize": 24})
                plt.imshow(images[j][0] * 0.5 + 0.5)
                plt.axis("off")

            histograms = [source_image_histograms[i], target_image_histograms[i], predicted_images_histograms[i]]
            if has_postprocess_columns:
                histograms += [post_processed_images_histograms[i]]

            for j in range(len(histograms)):
                idx += 1
                plt.subplot(num_images, num_columns, idx)
                plt.title(title[j+3] if i == 0 else "", fontdict={"fontsize": 24})
                plt.imshow(np.squeeze(np.clip(histograms[j] * 100., 0., 1.)))
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name, transparent=True)

        # cannot call show otherwise it flushes and empties the figure, sending to tensorboard
        # only a blank image... hence, let us just display the saved image
        display.display(figure)
        # plt.show()

        return figure

    def generate_images_from_dataset(self, dataset, step, num_images=None):
        if num_images is None:
            num_images = dataset.unbatch().cardinality()

        base_image_path = self.get_output_folder("test-images")

        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)

        for i, (source, target) in dataset.unbatch().take(num_images).batch(1).enumerate():
            image_path = os.sep.join([base_image_path, f"{i}_at_step_{step}.png"])
            fig = self.preview_generated_images_during_training([(source, target)], image_path, step)
            plt.close(fig)

        print(f"Generated {i + 1} images in the test-images folder.")

    def debug_discriminator_output(self, batch, image_path):
        # generates the fake image and the discriminations of the real and fake
        source_image, real_image = batch
        fake_image = self.proxy_generator(source_image, training=True)

        real_predicted = self.discriminator([real_image, source_image])
        fake_predicted = self.discriminator([fake_image, source_image])
        real_predicted = real_predicted[0]
        fake_predicted = fake_predicted[0]

        real_predicted = tf.math.sigmoid(real_predicted)
        fake_predicted = tf.math.sigmoid(fake_predicted)

        # finds the mean value of the patches (to display on the titles)
        real_predicted_mean = tf.reduce_mean(real_predicted)
        fake_predicted_mean = tf.reduce_mean(fake_predicted)

        # makes the patches have the same resolution as the real/fake images by repeating and tiling
        num_patches = tf.shape(real_predicted)[0]
        lower_bound_scaling_factor = IMG_SIZE // num_patches
        pad_before = (IMG_SIZE - num_patches * lower_bound_scaling_factor) // 2
        pad_after = (IMG_SIZE - num_patches * lower_bound_scaling_factor) - pad_before

        real_predicted = tf.repeat(tf.repeat(real_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        real_predicted = tf.pad(real_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        real_predicted = real_predicted[:, :, 0]
        fake_predicted = tf.repeat(tf.repeat(fake_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        fake_predicted = tf.pad(fake_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        fake_predicted = fake_predicted[:, :, 0]

        # gets rid of the batch dimension, as we have a batch of only one image
        real_image = real_image[0]
        fake_image = fake_image[0]
        source_image = source_image[0]

        # display the images: source / real / discr. real / fake / discr. fake
        plt.figure(figsize=(6 * 5, 6 * 1))
        plt.subplot(1, 6, 1)
        plt.title("Source", fontdict={"fontsize": 20})
        plt.imshow(source_image * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(1, 6, 2)
        plt.title("Target", fontdict={"fontsize": 20})
        plt.imshow(real_image * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(1, 6, 3)
        plt.title(tf.strings.join([
            "Discriminated target ",
            tf.strings.as_string(real_predicted_mean, precision=3)]).numpy().decode("UTF-8"), fontdict={"fontsize": 20})
        plt.imshow(real_predicted, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.subplot(1, 6, 4)
        plt.title("Generated", fontdict={"fontsize": 20})
        plt.imshow(fake_image * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(1, 6, 5)
        plt.title(tf.strings.reduce_join([
            "Discriminated generated ",
            tf.strings.as_string(fake_predicted_mean, precision=3)]).numpy().decode("UTF-8"), fontdict={"fontsize": 20})
        plt.imshow(fake_predicted, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.show()


class Pix2PixAugmentedModel(Pix2PixModel):
    def __init__(self, config):
        super().__init__(config)


class Pix2PixHistogramModel(Pix2PixAugmentedModel):
    def __init__(self, config):
        super().__init__(config)
        self.lambda_histogram = config.lambda_histogram
        if config.histo_loss == "hellinger":
            self.histo_loss = histogram.hellinger_loss
        elif config.histo_loss == "l1":
            self.histo_loss = histogram.l1_loss
        elif config.histo_loss == "l2":
            self.histo_loss = histogram.l2_loss
        else:
            raise Exception(f"Unrecognized histogram loss passed to the model: {config.histo_loss}")

    def generator_loss(self, fake_predicted, fake_image, real_image):
        real_histogram = histogram.calculate_rgbuv_histogram(real_image)
        fake_histogram = histogram.calculate_rgbuv_histogram(fake_image)
        histogram_loss = self.histo_loss(real_histogram, fake_histogram)

        total_loss, adversarial_loss, l1_loss = super().generator_loss(fake_predicted, fake_image, real_image)
        total_loss += self.lambda_histogram * histogram_loss

        return total_loss, adversarial_loss, l1_loss, histogram_loss

    def discriminator_loss(self, real_predicted, fake_predicted):
        return super().discriminator_loss(real_predicted, fake_predicted)

    def log_generator_loss(self, g_loss, step):
        _, _, _, histogram_loss = g_loss
        super().log_generator_loss(g_loss[:3], step)
        tf.summary.scalar("histogram_loss", histogram_loss, step=step)


class Pix2PixIndexedModel(Pix2PixModel):
    def __init__(self, config):
        super().__init__(config)
        self.lambda_segmentation = config.lambda_segmentation
        self.segmentation_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def create_generator(self):
        return UnetGenerator(1, MAX_PALETTE_SIZE, "softmax")

    def create_discriminator(self):
        return PatchDiscriminator(1)

    def generator_loss(self, fake_predicted, fake_image, real_image):
        segmentation_loss = self.segmentation_loss_object(real_image, fake_image)
        total_loss, adversarial_loss, l1_loss = super().generator_loss(fake_predicted, fake_image, real_image)
        total_loss += self.lambda_segmentation * segmentation_loss

        return total_loss, adversarial_loss, l1_loss, segmentation_loss

    def discriminator_loss(self, real_predicted, fake_predicted):
        return super().discriminator_loss(real_predicted, fake_predicted)

    def generate(self, batch):
        source_image, _, palette = batch
        fake_image_probabilities = self.generator(source_image, training=True)
        fake_image = tf.expand_dims(tf.argmax(fake_image_probabilities, axis=-1, output_type="int32"), -1)
        return fake_image

    def generate_with_probs(self, batch):
        source_image, _, palette = batch
        fake_image_probabilities = self.generator(source_image, training=True)
        fake_image = tf.expand_dims(tf.argmax(fake_image_probabilities, axis=-1, output_type="int32"), -1)
        return fake_image, fake_image_probabilities

    def train_step(self, batch, step, evaluate_steps, t):
        # batch: source_image, real_image, palette
        source_image, real_image, _ = batch
        batch_size = tf.shape(real_image)[0]

        real_image_one_hot = tf.reshape(tf.one_hot(real_image, MAX_PALETTE_SIZE, axis=-1),
                                        [batch_size, IMG_SIZE, IMG_SIZE, -1])
        with tf.GradientTape(persistent=True) as tape:
            fake_image, fake_image_probabilities = self.generate_with_probs(batch)

            real_predicted = self.discriminator([real_image, source_image], training=True)
            fake_predicted = self.discriminator([fake_image, source_image], training=True)

            g_loss = self.generator_loss(fake_predicted, fake_image_probabilities, real_image_one_hot)
            generator_total_loss = g_loss[0]

            d_loss = self.discriminator_loss(real_predicted, fake_predicted)
            discriminator_total_loss = d_loss[0]

        generator_gradients = tape.gradient(generator_total_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(discriminator_total_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            with tf.name_scope("generator"):
                self.log_generator_loss(g_loss, step // evaluate_steps)
            with tf.name_scope("discriminator"):
                self.log_discriminator_loss(d_loss, step // evaluate_steps)

    def log_generator_loss(self, g_loss, step):
        _, _, _, segmentation_loss = g_loss
        super().log_generator_loss(g_loss[:3], step)
        tf.summary.scalar("segmentation_loss", segmentation_loss, step=step)

    def preview_generated_images_during_training(self, examples, save_name, step):
        title = ["Input", "Target", "Generated"]
        num_images = len(examples)
        num_columns = len(title)

        if step is not None:
            title[-1] += f" ({step / 1000}k)"
        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))
        predicted_images = []

        for i, batch in enumerate(examples):
            source_image, target_image, palette = batch
            palette = palette[0]

            if i >= len(predicted_images):
                generated_image = self.generate(batch)
                predicted_images.append(generated_image)

            images = [source_image, target_image, predicted_images[i]]
            for j in range(num_columns):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                plt.title(title[j] if i == 0 else "", fontdict={"fontsize": 24})
                image = images[j][0]
                image = io_utils.indexed_to_rgba(image, palette)
                plt.imshow(image)
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)

        # cannot call show otherwise it flushes and empties the figure, sending to tensorboard
        # only a blank image... hence, let us just display the saved image
        display.display(figure)
        # plt.show()

        return figure

    def debug_discriminator_patches(self, batch_of_one):
        # generates the fake image and the discriminations of the real and fake
        source_image, real_image, palette = batch_of_one

        fake_image = self.proxy_generator(source_image, training=True)
        fake_image = tf.expand_dims(tf.argmax(fake_image, axis=-1, output_type="int32"), -1)

        real_predicted = self.discriminator([real_image, source_image])[0]
        fake_predicted = self.discriminator([fake_image, source_image])[0]

        real_predicted = tf.math.sigmoid(real_predicted)
        fake_predicted = tf.math.sigmoid(fake_predicted)

        # makes the patches have the same resolution as the real/fake images by repeating and tiling
        num_patches = tf.shape(real_predicted)[0]
        lower_bound_scaling_factor = IMG_SIZE // num_patches
        pad_before = (IMG_SIZE - num_patches * lower_bound_scaling_factor) // 2
        pad_after = (IMG_SIZE - num_patches * lower_bound_scaling_factor) - pad_before

        real_predicted = tf.repeat(tf.repeat(real_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        real_predicted = tf.pad(real_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        real_predicted = real_predicted[:, :, 0]
        fake_predicted = tf.repeat(tf.repeat(fake_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        fake_predicted = tf.pad(fake_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        fake_predicted = fake_predicted[:, :, 0]

        # gets rid of the batch dimension, as we have a batch of only one image
        real_image = real_image[0]
        fake_image = fake_image[0]
        palette = palette[0]

        # looks up the actual colors in the palette
        real_image = io_utils.indexed_to_rgba(real_image, palette)
        fake_image = io_utils.indexed_to_rgba(fake_image, palette)

        # display the images: real / discr. real / fake / discr. fake
        plt.figure(figsize=(6 * 4, 6 * 1))
        plt.subplot(1, 4, 1)
        plt.title("Target", fontdict={"fontsize": 20})
        plt.imshow(real_image, vmin=0, vmax=255)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Discriminated target", fontdict={"fontsize": 20})
        plt.imshow(real_predicted, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Generated", fontdict={"fontsize": 20})
        plt.imshow(fake_image, vmin=0, vmax=255)
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("Discriminated generated", fontdict={"fontsize": 20})
        plt.imshow(fake_predicted, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.show()

    def select_examples_for_evaluation(self, num_images, dataset):
        real_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, 4))
        fake_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, 4))
        dataset = dataset.unbatch().take(num_images).batch(1)

        for i, batch in dataset.enumerate():
            source_image, real_image, palette = batch
            fake_image = self.generate(batch)

            real_image = real_image[0]
            fake_image = fake_image[0]
            palette = palette[0]

            real_image = io_utils.indexed_to_rgba(real_image, palette)
            fake_image = io_utils.indexed_to_rgba(fake_image, palette)

            real_images[i] = real_image.numpy()
            fake_images[i] = fake_image.numpy()

        return real_images, fake_images
