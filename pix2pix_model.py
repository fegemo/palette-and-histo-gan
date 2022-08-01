import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from abc import abstractmethod

import histogram
import io_utils
from networks import *
from side2side_model import S2SModel


class Pix2PixModel(S2SModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name,
                 discriminator_type, generator_type, lambda_l1, lambda_histogram, **kwargs):
        super().__init__(train_ds, test_ds, model_name, architecture_name)

        default_kwargs = {"num_patches": 30}
        kwargs = {**default_kwargs, **kwargs}

        self.lambda_l1 = lambda_l1
        self.lambda_histogram = lambda_histogram

        self.generator = self.create_generator(generator_type)
        self.discriminator = self.create_discriminator(discriminator_type, **kwargs)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        generator_params = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in self.generator.trainable_weights])
        discriminator_params = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in self.discriminator.trainable_weights])

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
                                                             max_to_keep=5)

    @tf.function
    def train_step(self, batch, step, update_steps):
        source_image, real_image = batch

        with tf.GradientTape(persistent=True) as tape:
            fake_image = self.generator(source_image, training=True)

            real_predicted = self.discriminator([real_image, source_image], training=True)
            fake_predicted = self.discriminator([fake_image, source_image], training=True)

            g_loss = self.generator_loss(fake_predicted, fake_image, real_image, tf.cast(step, "float32")/10000.)
            generator_loss, generator_adversarial_loss, generator_l1_loss, generator_histogram_loss = g_loss

            d_loss = self.discriminator_loss(real_predicted, fake_predicted, fake_image, real_image)
            discriminator_loss, discriminator_real_loss, discriminator_fake_loss, _ = d_loss

        generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            with tf.name_scope("discriminator"):
                tf.summary.scalar("total_loss", discriminator_loss, step=step // update_steps)
                tf.summary.scalar("real_loss", discriminator_real_loss, step=step // update_steps)
                tf.summary.scalar("fake_loss", discriminator_fake_loss, step=step // update_steps)
            with tf.name_scope("generator"):
                tf.summary.scalar("total_loss", generator_loss, step=step // update_steps)
                tf.summary.scalar("adversarial_loss", generator_adversarial_loss, step=step // update_steps)
                tf.summary.scalar("l1_loss", generator_l1_loss, step=step // update_steps)
                tf.summary.scalar("histogram_loss", generator_histogram_loss, step=step // update_steps)

    def create_discriminator(self, discriminator_type, **kwargs):
        if discriminator_type == "patch":
            if "num_patches" not in kwargs:
                raise ValueError(
                    f"The 'num_patches' kw argument should have been passed to create_discriminator,"
                    f"but it was not. kwargs: {kwargs}")
            return PatchDiscriminator(kwargs["num_patches"])
        elif discriminator_type == "patch-resnet":
            return PatchResnetDiscriminator()
        elif discriminator_type == "deeper":
            return Deeper2x2PatchDiscriminator()
        elif discriminator_type == "u-net" or discriminator_type == "unet":
            return UnetDiscriminator()
        elif discriminator_type == "indexed-patch":
            return IndexedPatchDiscriminator(kwargs["num_patches"])
        elif discriminator_type == "indexed-patch-resnet":
            return IndexedPatchResnetDiscriminator()
        else:
            raise NotImplementedError(f"The {discriminator_type} type of discriminator has not been implemented")

    def create_generator(self, generator_type, **kwargs):
        if generator_type == "u-net" or generator_type == "unet":
            return UnetGenerator()
        elif generator_type == "atrous":
            raise NotImplementedError(f"The {generator_type} type of generator has not been implemented")
        elif generator_type == "indexed-unet":
            return IndexedUnetGenerator()
        else:
            raise NotImplementedError(f"The {generator_type} type of generator has not been implemented")

    def generator_loss(self, fake_predicted, fake_image, real_image, training_t):
        adversarial_loss = self.loss_object(tf.ones_like(fake_predicted), fake_predicted)
        l1_loss = tf.reduce_mean(tf.abs(real_image - fake_image))

        # real_histogram = histogram.calculate_rgbuv_histogram(real_image)
        # fake_histogram = histogram.calculate_rgbuv_histogram(fake_image)
        # histogram_loss = histogram.hellinger_loss(real_histogram, fake_histogram)
        histogram_loss = tf.constant(0., "float32")
        total_loss = adversarial_loss + (self.lambda_l1 * l1_loss) + (self.lambda_histogram * histogram_loss)

        return total_loss, adversarial_loss, l1_loss, histogram_loss

    def discriminator_loss(self, real_predicted, fake_predicted, fake_image, real_image):
        real_loss = self.loss_object(tf.ones_like(real_predicted), real_predicted)
        fake_loss = self.loss_object(tf.zeros_like(fake_predicted), fake_predicted)
        total_loss = fake_loss + real_loss

        return total_loss, real_loss, fake_loss

    def select_examples_for_visualization(self, num_examples=6):
        num_train_examples = num_examples // 2
        num_test_examples = num_examples - num_train_examples

        train_examples = self.train_ds.unbatch().take(num_train_examples).batch(1)
        test_examples = self.test_ds.unbatch().take(num_test_examples).batch(1)

        return list(test_examples.as_numpy_iterator()) + list(train_examples.as_numpy_iterator())

    def select_real_and_fake_images_for_fid(self, num_images, dataset):
        real_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        fake_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        dataset = dataset.unbatch().take(num_images).batch(1)

        for i, (source_image, real_image) in dataset.enumerate():
            fake_image = self.generator(source_image, training=True)
            real_images[i] = tf.squeeze(real_image).numpy()
            fake_images[i] = tf.squeeze(fake_image).numpy()

        return real_images, fake_images

    def evaluate_l1(self, real_images, fake_images):
        return tf.reduce_mean(tf.abs(fake_images - real_images))

    def generate_comparison(self, examples, save_name=None, step=None, predicted_images=None):
        # invoca o gerador e mostra a imagem de entrada, sua imagem objetivo e a imagem gerada
        title = ["Input", "Target", "Generated"]
        num_images = len(examples)
        num_columns = len(title)

        if step is not None:
            title[-1] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))

        if predicted_images is None:
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

    def show_discriminated_image(self, batch_of_one):
        # generates the fake image and the discriminations of the real and fake
        source_image, real_image = batch_of_one
        fake_image = self.generator(source_image, training=True)

        real_predicted = self.discriminator([real_image, source_image])
        fake_predicted = self.discriminator([fake_image, source_image])
        desired_fake_prediction = self.calculate_desired_fake_prediction(fake_image, real_image, fake_predicted)
        real_predicted = real_predicted[0]
        fake_predicted = fake_predicted[0]

        real_predicted = tf.math.sigmoid(real_predicted)
        fake_predicted = tf.math.sigmoid(fake_predicted)

        # finds the mean value of the patches (to display on the titles)
        real_predicted_mean = tf.reduce_mean(real_predicted)
        fake_predicted_mean = tf.reduce_mean(fake_predicted)
        desired_fake_prediction_mean = tf.reduce_mean(desired_fake_prediction)

        # makes the patches have the same resolution as the real/fake images by repeating and tiling
        num_patches = tf.shape(real_predicted)[0]
        lower_bound_scaling_factor = IMG_SIZE // num_patches
        pad_before = (IMG_SIZE - num_patches * lower_bound_scaling_factor) // 2
        pad_after = (IMG_SIZE - num_patches * lower_bound_scaling_factor) - pad_before

        real_predicted = tf.repeat(tf.repeat(real_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        real_predicted = tf.pad(real_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        fake_predicted = tf.repeat(tf.repeat(fake_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        fake_predicted = tf.pad(fake_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])

        # gets rid of the batch dimension, as we have a batch of only one image
        real_image = real_image[0]
        fake_image = fake_image[0]
        source_image = source_image[0]
        desired_fake_prediction = desired_fake_prediction[0]

        # display the images: source / real / discr. real / fake / discr. fake / desired discr. fake
        plt.figure(figsize=(6 * 6, 6 * 1))
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
            "Discriminated label ",
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

        plt.subplot(1, 6, 6)
        plt.title(tf.strings.reduce_join([
            "Desired disc. output ",
            tf.strings.as_string(desired_fake_prediction_mean, precision=3)]).numpy().decode("UTF-8"),
                  fontdict={"fontsize": 20})
        plt.imshow(desired_fake_prediction, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.show()


class Pix2PixIndexedModel(Pix2PixModel):
    def __init__(self, train_ds, test_ds, model_name, architecture_name,
                 discriminator_type, generator_type, lambda_l1=100., lambda_segmentation=0.5, **kwargs):
        super().__init__(train_ds, test_ds, model_name, architecture_name,
                         discriminator_type, generator_type, lambda_l1, **kwargs)
        self.lambda_segmentation = lambda_segmentation
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def generator_loss(self, fake_predicted, fake_image, real_image):
        adversarial_loss = self.loss_object(tf.ones_like(fake_predicted), fake_predicted)
        l1_loss = tf.reduce_mean(tf.abs(real_image - fake_image))
        segmentation_loss = self.generator_loss_object(real_image, fake_image)
        total_loss = adversarial_loss + (self.lambda_l1 * l1_loss) + (self.lambda_segmentation * segmentation_loss)

        return total_loss, adversarial_loss, l1_loss, segmentation_loss

    def discriminator_loss(self, real_predicted, fake_predicted):
        real_loss = self.loss_object(tf.ones_like(real_predicted), real_predicted)
        fake_loss = self.loss_object(tf.zeros_like(fake_predicted), fake_predicted)
        total_loss = real_loss + fake_loss

        return total_loss, real_loss, fake_loss

    def train_step(self, batch, step, update_steps):
        source_image, real_image, palette = batch
        batch_size = tf.shape(source_image)[0]

        real_image_one_hot = tf.reshape(tf.one_hot(real_image, MAX_PALETTE_SIZE, axis=-1), [batch_size, IMG_SIZE, IMG_SIZE, -1])

        with tf.GradientTape(persistent=True) as tape:
            fake_image = self.generator(source_image, training=True)
            fake_image_for_discriminator = tf.expand_dims(tf.argmax(fake_image, axis=-1, output_type="int32"), -1)

            real_predicted = self.discriminator([real_image, source_image], training=True)
            fake_predicted = self.discriminator([fake_image_for_discriminator, source_image], training=True)

            generator_loss, generator_adversarial_loss, generator_l1_loss, generator_segmentation_loss = \
                self.generator_loss(fake_predicted, fake_image, real_image_one_hot)
            discriminator_loss, discriminator_real_loss, discriminator_fake_loss = self.discriminator_loss(
                real_predicted, fake_predicted)

        discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            with tf.name_scope("discriminator"):
                tf.summary.scalar("total_loss", discriminator_loss, step=step // update_steps)
                tf.summary.scalar("real_loss", discriminator_real_loss, step=step // update_steps)
                tf.summary.scalar("fake_loss", discriminator_fake_loss, step=step // update_steps)
            with tf.name_scope("generator"):
                tf.summary.scalar("total_loss", generator_loss, step=step // update_steps)
                tf.summary.scalar("adversarial_loss", generator_adversarial_loss, step=step // update_steps)
                tf.summary.scalar("l1_loss", generator_l1_loss, step=step // update_steps)
                tf.summary.scalar("segmentation_loss", generator_segmentation_loss, step=step // update_steps)

    def select_examples_for_visualization(self, num_examples=6):
        num_train_examples = num_examples // 2
        num_test_examples = num_examples - num_train_examples

        test_examples = self.test_ds.unbatch().take(num_test_examples).batch(1)
        train_examples = self.train_ds.unbatch().take(num_train_examples).batch(1)
        return list(test_examples.as_numpy_iterator()) + list(train_examples.as_numpy_iterator())

    def generate_comparison(self, examples, save_name=None, step=None, predicted_images=None):
        # invoca o gerador e mostra a imagem de entrada, sua imagem objetivo e a imagem gerada
        num_images = len(examples)
        num_columns = 3

        title = ["Input", "Target", "Generated"]
        if step is not None:
            title[-1] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))

        if predicted_images is None:
            predicted_images = []

        for i, (source_image, target_image, palette) in enumerate(examples):
            palette = palette[0]

            if i >= len(predicted_images):
                generated_image = self.generator(source_image, training=True)
                # tf.print("tf.shape(generated_image)", tf.shape(generated_image))
                generated_image = tf.expand_dims(tf.argmax(generated_image, axis=-1, output_type="int32"), -1)
                # tf.print("tf.shape(generated_image) after argmax", tf.shape(generated_image))
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

    def show_discriminated_image(self, batch_of_one):
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
        fake_predicted = tf.repeat(tf.repeat(fake_predicted, lower_bound_scaling_factor, axis=0),
                                   lower_bound_scaling_factor, axis=1)
        fake_predicted = tf.pad(fake_predicted, [[pad_before, pad_after], [pad_before, pad_after], [0, 0]])

        # gets rid of the batch dimension, as we have a batch of only one image
        real_image = real_image[0]
        fake_image = fake_image[0]
        palette = palette[0]

        # looks up the actual colors in the palette
        real_image = io_utils.indexed_to_rgba(real_image, palette)
        fake_image = io_utils.indexed_to_rgba(fake_image, palette)

        # display the images: real / discr. real / fake / discr. fake
        figure = plt.figure(figsize=(6 * 4, 6 * 1))
        plt.subplot(1, 4, 1)
        plt.title("Label", fontdict={"fontsize": 20})
        plt.imshow(real_image, vmin=0, vmax=255)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Discriminated label", fontdict={"fontsize": 20})
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

    def select_real_and_fake_images_for_fid(self, num_images, dataset):
        real_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        fake_images = np.ndarray((num_images, IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS))
        dataset = dataset.unbatch().take(num_images).batch(1)

        for i, (source_image, real_image, palette) in dataset.enumerate():
            fake_image = self.generator(source_image, training=True)
            fake_image = tf.expand_dims(tf.argmax(fake_image, axis=-1, output_type="int32"), -1)

            real_image = real_image[0]
            fake_image = fake_image[0]
            palette = palette[0]

            real_image = io_utils.indexed_to_rgba(real_image, palette)
            fake_image = io_utils.indexed_to_rgba(fake_image, palette)

            real_images[i] = real_image.numpy()
            fake_images[i] = fake_image.numpy()

        return real_images, fake_images
