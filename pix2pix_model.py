import numpy as np
from IPython import display
from matplotlib import pyplot as plt

import histogram
import io_utils
from networks import *
from side2side_model import S2SModel


class Pix2PixModel(S2SModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name, lambda_l1):
        super().__init__(train_ds, test_ds, model_name, architecture_name)

        self.lambda_l1 = lambda_l1

        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        generator_params = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in self.generator.trainable_weights])
        discriminator_params = tf.reduce_sum(
            [tf.reduce_prod(v.get_shape()) for v in self.discriminator.trainable_weights])

        print(f"Generator: {self.generator.name} with {generator_params:,} parameters")
        print(f"Discriminator: {self.discriminator.name} with {discriminator_params:,} parameters")

        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir,
                                                             max_to_keep=1)

    def create_generator(self):
        return UnetGenerator(4, 4, "tanh")

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
        return self.generator(source_image, training=True)

    @tf.function
    def train_step(self, batch, step, update_steps):
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
                self.log_generator_loss(g_loss, step // update_steps)
            with tf.name_scope("discriminator"):
                self.log_discriminator_loss(d_loss, step // update_steps)

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

    def select_examples_for_visualization(self, number_of_examples=6):
        num_train_examples = number_of_examples // 2
        num_test_examples = number_of_examples - num_train_examples

        train_examples = self.train_ds.unbatch().take(num_train_examples).batch(1)
        test_examples = self.test_ds.unbatch().take(num_test_examples).batch(1)

        return list(test_examples.as_numpy_iterator()) + list(train_examples.as_numpy_iterator())

    def select_examples_for_evaluation(self, num_images, dataset):
        real_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, 4))
        fake_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, 4))
        dataset = dataset.unbatch().take(num_images).batch(1)

        for i, (source_image, real_image) in dataset.enumerate():
            fake_image = self.generator(source_image, training=True)
            real_images[i] = real_image[0].numpy()
            fake_images[i] = fake_image[0].numpy()

        return real_images, fake_images

    def evaluate_l1(self, real_images, fake_images):
        return tf.reduce_mean(tf.abs(fake_images - real_images))

    def preview_generated_images_during_training(self, examples, save_name, step):
        title = ["Input", "Target", "Generated"]
        num_images = len(examples)
        num_columns = len(title)

        if step is not None:
            title[-1] += f" ({step / 1000}k)"
        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))
        predicted_images = []

        for i, (source_image, target_image) in enumerate(examples):
            if i >= len(predicted_images):
                predicted_images.append(self.generator(source_image, training=True))

            images = [source_image, target_image, predicted_images[i]]
            for j in range(num_columns):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                plt.title(title[j] if i == 0 else "", fontdict={"fontsize": 24})
                plt.imshow(images[j][0] * 0.5 + 0.5)
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
        source_image, real_image = batch_of_one
        fake_image = self.generator(source_image, training=True)

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
    def __init__(self, train_ds, test_ds, model_name, architecture_name, lambda_l1):
        super().__init__(train_ds, test_ds, model_name, architecture_name, lambda_l1)


class Pix2PixHistogramModel(Pix2PixAugmentedModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name, lambda_l1, lambda_histogram):
        super().__init__(train_ds, test_ds, model_name, architecture_name, lambda_l1)
        self.lambda_histogram = lambda_histogram

    def generator_loss(self, fake_predicted, fake_image, real_image):
        real_histogram = histogram.calculate_rgbuv_histogram(real_image)
        fake_histogram = histogram.calculate_rgbuv_histogram(fake_image)
        histogram_loss = histogram.hellinger_loss(real_histogram, fake_histogram)

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
    def __init__(self, train_ds, test_ds, model_name, architecture_name, lambda_segmentation=0.5):
        super().__init__(train_ds, test_ds, model_name, architecture_name, 0.)
        self.lambda_segmentation = lambda_segmentation
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

    def train_step(self, batch, step, update_steps):
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
                self.log_generator_loss(g_loss, step // update_steps)
            with tf.name_scope("discriminator"):
                self.log_discriminator_loss(d_loss, step // update_steps)

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

        fake_image = self.generator(source_image, training=True)
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
