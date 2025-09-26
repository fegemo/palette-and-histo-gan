import os

import tensorflow_io as tfio
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

import histogram
import io_utils
from keras_utils import NParamsSupplier
from networks import *
from side2side_model import S2SModel


class PostProcessGenerator(tf.keras.Model):
    def __init__(self, real_generator, post_process_type):
        super().__init__()
        self.real_generator = real_generator
        self.post_process_type = post_process_type

    def __call__(self, batch, **kwargs):
        fake_image = self.real_generator(batch, **kwargs)
        palette = io_utils.batch_extract_palette(batch, )
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

        self.gen_supplier = NParamsSupplier(2 if config.palette_quantization else 1)
        self.lambda_l1 = config.lambda_l1
        self.lambda_palette = config.lambda_palette
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def create_generator(self):
        config = self.config
        real_generator = UnetGenerator(config.image_size, config.inner_channels, config.output_channels,
                                       "tanh", config.palette_quantization, config.temperature)
        if self.config.post_process is not None and self.config.post_process != "none":
            self.proxy_generator = PostProcessGenerator(real_generator, self.config.post_process)
        else:
            self.proxy_generator = real_generator
        return real_generator

    def create_discriminator(self):
        config = self.config
        return PatchDiscriminator(config.image_size, config.inner_channels)

    def get_annealing_layers(self):
        return [self.generator.quantization] if self.config.palette_quantization else []

    def generator_loss(self, fake_predicted, fake_image, real_image, palette, temperature):
        adversarial_loss = self.loss_object(tf.ones_like(fake_predicted), fake_predicted)
        l1_loss = tf.reduce_mean(tf.abs(real_image - fake_image))
        palette_loss = self.calculate_palette_loss(fake_image, palette, temperature)

        total_loss = adversarial_loss + \
                     self.lambda_l1 * l1_loss + \
                     self.lambda_palette * palette_loss

        return total_loss, adversarial_loss, l1_loss, palette_loss

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

        # updates the annealing scheduler to get the new temperature
        temperature = self.annealing_scheduler.update(t)

        # potentially extract the palette, in case we are using palette quantization
        palette = self.extract_palette(source_image)

        with tf.GradientTape(persistent=True) as tape:
            fake_image = self.generator(self.gen_supplier(source_image, palette), training=True)

            real_predicted = self.discriminator([real_image, source_image], training=True)
            fake_predicted = self.discriminator([fake_image, source_image], training=True)

            g_loss = self.generator_loss(fake_predicted, fake_image, real_image, palette, temperature)
            generator_total_loss = g_loss[0]

            d_loss = self.discriminator_loss(real_predicted, fake_predicted)
            discriminator_total_loss = d_loss[0]

        generator_gradients = tape.gradient(generator_total_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(discriminator_total_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        del tape

        with self.summary_writer.as_default():
            with tf.name_scope("generator"):
                self.log_generator_loss(g_loss, step // evaluate_steps)
            with tf.name_scope("discriminator"):
                self.log_discriminator_loss(d_loss, step // evaluate_steps)

    def log_generator_loss(self, g_loss, step):
        total_loss, adversarial_loss, l1_loss, palette_loss = g_loss
        tf.summary.scalar("total_loss", total_loss, step=step)
        tf.summary.scalar("adversarial_loss", adversarial_loss, step=step)
        tf.summary.scalar("l1_loss", l1_loss, step=step)
        tf.summary.scalar("palette_loss", palette_loss, step=step)

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
        c = self.config
        real_images = np.ndarray((num_images, c.image_size, c.image_size, 4))
        fake_images = np.ndarray((num_images, c.image_size, c.image_size, 4))
        dataset = dataset.unbatch().take(num_images).batch(1)

        for i, (source_image, real_image) in dataset.enumerate():
            fake_image = self.proxy_generator(source_image, training=True)
            real_images[i] = real_image[0].numpy()
            fake_images[i] = fake_image[0].numpy()

        return real_images, fake_images

    def initialize_random_examples_for_evaluation(self, train_ds, test_ds, num_images):
        def initialize_random_examples_from_dataset(dataset):
            source_images, target_images = next(iter(dataset.unbatch().batch(num_images).take(1)))
            return source_images, target_images

        return dict({
            "train": initialize_random_examples_from_dataset(train_ds),
            "test": initialize_random_examples_from_dataset(test_ds.shuffle(self.config.test_size))
        })

    def generate_images_for_evaluation(self, example_indices_for_evaluation):
        generator = self.generator
        def generate_images_from_dataset(dataset_name):
            source_images, target_images = example_indices_for_evaluation[dataset_name]
            palettes = self.extract_palette(source_images)
            fake_images = generator(self.gen_supplier(source_images, palettes), training=False)
            return target_images, fake_images

        return dict({
            "train": generate_images_from_dataset("train"),
            "test": generate_images_from_dataset("test")
        })

    def evaluate_l1(self, real_images, fake_images):
        # show_grid_of_images([real_images[:4], fake_images[:4]], ["Real", "Fake"])
        return tf.reduce_mean(tf.abs(fake_images - real_images))

    def preview_generated_images_during_training(self, examples, save_name, step):
        has_postprocess_columns = self.config.post_process != "none"
        palette_quantization = self.config.palette_quantization
        title = ["Input", "Target", "Generated", "Input histo", "Target histo", "Generated histo"]
        if has_postprocess_columns:
            title = title[:3] + ["Post-processed"] + title[3:] + ["Pstpcssd histo"]
        elif palette_quantization:
            title = title[:3] + ["Generated (t=0)"] + title[3:] + ["Gen (t=0) histo"]
            temperature = self.get_annealing_layers()[0].temperature.numpy()
            title[2] = f"Gener. (t={float(temperature):.2f})"
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
        zero_temperature_predicted_histograms = []

        for i, (source_image, target_image) in enumerate(examples):
            source_image = tf.convert_to_tensor(source_image)
            palette = self.extract_palette(source_image)
            if i >= len(predicted_images):
                predicted_image = self.generator(self.gen_supplier(source_image, palette), training=True)
                predicted_images.append(predicted_image)
                predicted_images_histograms.append(
                    histogram.calculate_rgbuv_histogram(predicted_image))

            images = [source_image, target_image, predicted_images[i]]
            if has_postprocess_columns:
                post_processed_image = self.proxy_generator(source_image, training=True)
                images += [post_processed_image]
                post_processed_images_histograms.append(
                    histogram.calculate_rgbuv_histogram(post_processed_image))
            elif palette_quantization:
                predicted_image_zero_temp = self.generator(self.gen_supplier(source_image, palette),
                                                           training=False)
                images += [predicted_image_zero_temp]
                zero_temperature_predicted_histograms.append(
                    histogram.calculate_rgbuv_histogram(predicted_image_zero_temp))

            for j in range(len(images)):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                plt.title(title[j] if i == 0 else "", fontdict={"fontsize": 24})
                plt.imshow(images[j][0] * 0.5 + 0.5)
                plt.axis("off")

            histograms = [source_image_histograms[i], target_image_histograms[i], predicted_images_histograms[i]]
            if has_postprocess_columns:
                histograms += [post_processed_images_histograms[i]]
            elif palette_quantization:
                histograms += [zero_temperature_predicted_histograms[i]]

            first_histogram_column_index = len(title) // 2
            for j in range(len(histograms)):
                idx += 1
                plt.subplot(num_images, num_columns, idx)
                plt.title(title[j+first_histogram_column_index] if i == 0 else "", fontdict={"fontsize": 24})
                plt.imshow(np.squeeze(np.clip(histograms[j] * 100., 0., 1.)))
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name, transparent=True)

        return figure

    def generate_images_from_dataset(self, dataset, step, num_images=None):
        if num_images is None:
            num_images = dataset.unbatch().cardinality()

        base_image_path = self.get_output_folder("test-images")

        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)

        for i, batch in dataset.unbatch().take(num_images).batch(1).enumerate():
            image_path = os.sep.join([base_image_path, f"{i}_at_step_{step}.png"])
            fig = self.preview_generated_images_during_training([[*batch]], image_path, step)
            plt.close(fig)

        print(f"Generated {i + 1} images in the test-images folder.")

    def debug_discriminator_output(self, batch, image_path):
        c = self.config
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
        lower_bound_scaling_factor = c.image_size // num_patches
        pad_before = (c.image_size - num_patches * lower_bound_scaling_factor) // 2
        pad_after = (c.image_size - num_patches * lower_bound_scaling_factor) - pad_before

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
        gen = UnetGenerator(self.config.image_size, self.config.input_channels,
                             self.config.output_channels, "softmax")
        return gen

    def create_discriminator(self):
        return PatchDiscriminator(self.config.image_size, self.config.input_channels)

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
        return fake_image, fake_image_probabilities

    def generate_rgba(self, batch):
        _, _, palette = batch
        fake_image, _ = self.generate(batch)
        fake_image = io_utils.batch_indexed_to_rgba(fake_image, palette)
        return fake_image

    @tf.function
    def train_step(self, batch, step, evaluate_steps, t):
        c = self.config
        # batch: source_image, real_image, palette
        source_image, real_image, _ = batch
        batch_size = tf.shape(real_image)[0]

        real_image_one_hot = tf.reshape(tf.one_hot(real_image, c.max_palette_size, axis=-1),
                                        [batch_size, c.image_size, c.image_size, c.max_palette_size])
        with tf.GradientTape(persistent=True) as tape:
            fake_image, fake_image_probabilities = self.generate(batch)

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
            if i >= len(predicted_images):
                # if i == 0:
                #     print("PPPPPP batch inside preview bf generate_rgba", type(batch))
                #     print("batch[0][0].shape", batch[0][0].shape)
                #     print("tf.reduce_max(batch[0][0])", tf.reduce_max(batch[0][0]))
                #     print("tf.reduce_max(batch[1][0])", tf.reduce_max(batch[1][0]))
                #     print("tf.reduce_max(batch[2][0])", tf.reduce_max(batch[2][0]))
                #     show_single_image(batch[0][0], "batch[0][0] inside preview")
                #     show_single_image(batch[1][0], "batch[1][0] inside preview")
                generated_image = self.generate_rgba(batch)
                # if i == 0:
                #     print("tf.reduce_max(generated_image[0])", tf.reduce_max(generated_image[0]))
                #     show_single_image(generated_image[0]*0.5+0.5, "generated_image[0] inside preview")
                predicted_images.append(generated_image)

            source_image, target_image, palette = batch
            source_image = io_utils.batch_indexed_to_rgba(source_image, palette)
            target_image = io_utils.batch_indexed_to_rgba(target_image, palette)
            images = [source_image, target_image, predicted_images[i]]
            for j in range(num_columns):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                plt.title(title[j] if i == 0 else "", fontdict={"fontsize": 24})
                image = tf.squeeze(images[j]) * 0.5 + 0.5
                plt.imshow(image)
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)

        return figure

    def debug_discriminator_patches(self, batch_of_one):
        c = self.config
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
        lower_bound_scaling_factor = c.image_size // num_patches
        pad_before = (c.image_size - num_patches * lower_bound_scaling_factor) // 2
        pad_after = (c.image_size - num_patches * lower_bound_scaling_factor) - pad_before

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
        image_size = self.config.image_size
        real_images = np.ndarray((num_images, image_size, image_size, 4))
        fake_images = np.ndarray((num_images, image_size, image_size, 4))
        dataset = dataset.unbatch().take(num_images).batch(1)

        for i, batch in dataset.enumerate():
            source_image, real_image, palette = batch
            fake_image = self.generate_rgba(batch)

            real_image = real_image[0]
            fake_image = fake_image[0]
            palette = palette[0]

            real_image = io_utils.indexed_to_rgba(real_image, palette)
            fake_image = io_utils.indexed_to_rgba(fake_image, palette)

            real_images[i] = real_image.numpy()
            fake_images[i] = fake_image.numpy()

        return real_images, fake_images

    def generate_images_for_evaluation(self, example_indices_for_evaluation):
        def generate_images_from_dataset(dataset_name):
            batch = example_indices_for_evaluation[dataset_name]
            # print("EEEEEE batch inside gen_evaluate bf generate_rgba", type(batch))
            # print("batch[0][0].shape", batch[0][0].shape)
            # print("tf.reduce_max(batch[0][0])", tf.reduce_max(batch[0][0]))
            # print("tf.reduce_max(batch[1][0])", tf.reduce_max(batch[1][0]))
            # print("tf.reduce_max(batch[2][0])", tf.reduce_max(batch[2][0]))
            # show_single_image(batch[0][0], "batch[0][0] inside gen_evaluate")
            # show_single_image(batch[1][0], "batch[1][0] inside gen_evaluate")

            fake_images = self.generate_rgba(batch)
            # print("tf.reduce_max(fake_images[0])", tf.reduce_max(fake_images[0]))
            # show_single_image(fake_images[0]*0.5+0.5, "fake_images[0] inside gen_evaluate")
            source_image, target_image, palette = batch
            real_images = io_utils.batch_indexed_to_rgba(target_image, palette)
            return real_images, fake_images

        return dict({
            "train": generate_images_from_dataset("train"),
            "test": generate_images_from_dataset("test")
        })





def show_single_image(image, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def show_grid_of_images(image_columns, titles):
    num_columns = len(image_columns)
    num_rows = len(image_columns[0])
    plt.figure(figsize=(4 * num_columns, 4 * num_rows))
    for i in range(num_rows):
        for j in range(num_columns):
            idx = i * num_columns + j + 1
            plt.subplot(num_rows, num_columns, idx)
            if i == 0:
                plt.title(titles[j], fontdict={"fontsize": 24})
            plt.imshow(np.clip(image_columns[j][i] * 0.5 + 0.5, 0., 1.))
            plt.axis("off")
    plt.tight_layout()
    plt.show()