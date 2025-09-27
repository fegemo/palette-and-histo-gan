from abc import abstractmethod, ABC

import tensorflow as tf

class NParamsSupplier:
    def __init__(self, supply_first_n_params, delistify=False):
        self.n = supply_first_n_params
        self.select_params = lambda args: args[0] if supply_first_n_params == 1 and delistify else args[0:supply_first_n_params]

    def __call__(self, *args, **kwargs):
        return self.select_params(args)


def count_network_parameters(network):
    return tf.reduce_sum([tf.reduce_prod(v.shape) for v in network.trainable_weights])


# ------ layer in which each palette can have a different number of colors ------
# this refrains from using vectorized operations, which make it slower for longer batches
class DifferentiablePaletteQuantization(tf.keras.layers.Layer):
    def __init__(self, initial_temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = self.add_weight(shape=(), name="temperature", trainable=False,
                                           initializer=tf.constant_initializer(initial_temperature))

    def call(self, inputs, training=None):
        """
        Returns the image with its colors quantized to the palette. During training, it is done with
        a soft assignment using the softmax function with a temperature. During inference, it is done
        with a hard assignment, using the closest color in the palette (hence, losing differentiability).

        :param inputs: Tuple of (images, palettes), with shapes:
            - Option 1: [b, h, w, c] and [b, (k), c] (original format)
            - Option 2: [b, num_domains, h, w, c] and [b, k, c] (format with full example, but does not support
                ragged palettes)
        :param training: True if training (uses soft assignment through softmax with temperature)
            or False otherwise (uses hard assignment, losing differentiability)
        :return: Quantized images: Tensor of shape matching input images shape
        """
        images, palettes = inputs

        # Determine input format based on rank
        if images.shape.rank == 4:
            # single image per item in the batch format: [b, h, w, c]
            return self.quantize_images(images, palettes, training)
        elif images.shape.rank == 5:
            # full example per item in the batch format: [b, num_domains, h, w, c]
            batch_size, num_domains, image_size, channels = (tf.shape(images)[0], tf.shape(images)[1],
                                                             tf.shape(images)[2], tf.shape(images)[-1])

            # Repeat the palette for each domain
            palettes_expanded = tf.tile(tf.expand_dims(palettes, 1), [1, num_domains, 1, 1])
            # palettes_expanded (shape=[b, num_domains, k, c])

            # Reshape to combine batch and domains dimensions
            images_reshaped = tf.reshape(images, [batch_size * num_domains,
                                                  image_size, image_size, channels])
            palettes_reshaped = tf.reshape(palettes_expanded, [batch_size * num_domains, -1, channels])
            # images_reshaped (shape=[b * num_domains, h, w, c])
            # palettes_reshaped (shape=[b * num_domains, k, c])

            # Quantize all images
            quantized = self.quantize_images(images_reshaped, palettes_reshaped, training)

            # Reshape back to original format
            return tf.reshape(quantized, [batch_size, num_domains] + quantized.shape[1:].as_list())
        else:
            raise ValueError(f"Unsupported input rank: {images.shape.rank}. Expected 4 or 5.")

    def quantize_images(self, images, palettes, training):
        """Helper function to quantize images with the given palettes."""

        # Process each (image, palette) pair independently
        def quantize_single_image(args):
            # img: [H, W, C]
            # palette: [K, C] (variable K)
            img, palette = args
            distances = tf.reduce_sum(
                (tf.expand_dims(img, -2) - tf.expand_dims(palette, 0)) ** 2,
                axis=-1
            )  # [H, W, K]

            if training:
                temperature = tf.maximum(self.temperature, 1e-8)
                weights = tf.nn.softmax(-distances / temperature, axis=-1)
                return tf.einsum('...k,kc->...c', weights, palette)  # [H, W, C]
            else:
                indices = tf.argmin(distances, axis=-1)
                return tf.gather(palette, indices)

        # there is no easy way to vectorize this operation, so we use map_fn to
        # process each <image,palette> pair independently
        images_shape = images.shape
        image_size, channels = images_shape[-3], images_shape[-1]
        return tf.map_fn(
            fn=quantize_single_image,
            elems=(images, palettes),
            fn_output_signature=tf.TensorSpec([image_size, image_size, channels], tf.float32),
            infer_shape=False
        )

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class AnnealingScheduler(ABC):
    def __init__(self, annealing_layers=None, min_temperature=0.015):
        if annealing_layers is None:
            annealing_layers = []
        self.annealing_layers = annealing_layers
        self.min_temperature = min_temperature

    def update(self, t):
        new_temperature = tf.maximum(self.get_value(t), self.min_temperature)
        for l in self.annealing_layers:
            l.temperature.assign(new_temperature)
        return new_temperature

    @abstractmethod
    def get_value(self, t):
        pass


class LinearAnnealingScheduler(AnnealingScheduler):
    def __init__(self, initial_temperature, annealing_layers):
        super().__init__(annealing_layers)
        self.initial_temperature = initial_temperature

    def get_value(self, t):
        return tf.maximum(0.0, (1.0 - t) * self.initial_temperature)



class CosineAnnealingScheduler(AnnealingScheduler):
    """
    Cosine annealing scheduler for temperature annealing. It reduces the temperature from the initial value to 0
    following a cosine curve with an amplitude that decreases as it approaches zero.
    Formula: https://www.google.com.br/search?q=f%28x%29%3D%281-x%29%2810%2B3+cos+%28x%E2%8B%8537.7%29%29&sca_esv=37dc570b99b93daa&hl=pt-BR&sxsrf=AE3TifO9WPcA8DRjJyVbKQiHxldxLgaPxQ%3A1755614259740&ei=M4ykaIfoLNyG1sQP87G1kQs&ved=0ahUKEwjHzcHijJePAxVcg5UCHfNYLbIQ4dUDCBA&uact=5&oq=f%28x%29%3D%281-x%29%2810%2B3+cos+%28x%E2%8B%8537.7%29%29&gs_lp=Egxnd3Mtd2l6LXNlcnAiH2YoeCk9KDEteCkoMTArMyBjb3MgKHjii4UzNy43KSkyBRAAGO8FMggQABiABBiiBDIFEAAY7wUyBRAAGO8FSMgyUOESWNsocAJ4AZABAJgBfqAB5wOqAQMwLjS4AQPIAQD4AQGYAgagAvsDwgIKEAAYsAMY1gQYR5gDAIgGAZAGCJIHAzIuNKAH0gmyBwMwLjS4B_EDwgcFMC40LjLIBxA&sclient=gws-wiz-serp
    """
    def __init__(self, initial_temperature, cycles, annealing_layers):
        super().__init__(annealing_layers)
        self.initial_temperature = initial_temperature
        self.cycles = cycles
        self.amplitude_variance = 0.15
        self.FULL_CYCLE = 2 * 3.14159

    def get_value(self, t):
        t0 = self.initial_temperature
        sigma = self.amplitude_variance
        lamb = self.cycles
        CYC = self.FULL_CYCLE
        return (1 - t) * ((1 - sigma) * t0 + (t0 * sigma) * tf.math.cos(t * lamb * CYC))

class ExpCosineAnnealingSchedule(CosineAnnealingScheduler):
    def __init__(self, initial_temperature, cycles, annealing_layers):
        super().__init__(initial_temperature, cycles, annealing_layers)

    def get_value(self, t):
        t0 = self.initial_temperature
        sigma = self.amplitude_variance
        lamb = self.cycles
        CYC = self.FULL_CYCLE
        return tf.math.pow(1 - t, 4) * ((1 - sigma) * t0 + (t0 * sigma) * tf.math.cos(t * lamb * CYC))


class NoopAnnealingScheduler(AnnealingScheduler):
    def get_value(self, t):
        return 1.0


