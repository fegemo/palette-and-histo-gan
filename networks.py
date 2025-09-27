import tensorflow as tf
from tensorflow.keras import layers

import keras_utils


def unet_downsample(filters, size, apply_batchnorm=True, init=tf.random_normal_initializer(0., 0.02)):
    result = tf.keras.Sequential()

    result.add(layers.Conv2D(
        filters,
        size,
        strides=2,
        padding="same",
        kernel_initializer=init,
        use_bias=False))
    if apply_batchnorm:
        result.add(layers.GroupNormalization(groups=filters))
    result.add(layers.LeakyReLU())

    return result


def unet_upsample(filters, size, apply_dropout=False, init=tf.random_normal_initializer(0., 0.02)):
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2, padding="same", kernel_initializer=init, use_bias=False))

    result.add(layers.GroupNormalization(groups=filters))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result


def PatchDiscriminator(image_size, inner_channels):
    initializer = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[image_size, image_size, inner_channels], name="source_image")
    target_image = layers.Input(shape=[image_size, image_size, inner_channels], name="target_image")

    x = layers.concatenate([target_image, source_image])                           # (batch_size, 64, 64, channels*2)
    down = unet_downsample(64, 4, False)(x)              # (batch_size, 32, 32,         64)
    last = layers.Conv2D(1, 4, padding="same",                    # (batch_size, 32, 32,          1)
                         kernel_initializer=initializer)(down)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=last, name="patch-disc")


def UnetGenerator(image_size, inner_channels, output_channels, last_activation,
                  palette_quantization=False, temperature=1.0):
    init = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[image_size, image_size, inner_channels])               # (batch_size, 64, 64, 4/3/max_palette_size)

    down_stack = [
        unet_downsample( 64, 4, apply_batchnorm=False, init=init),                  # (batch_size, 32, 32,   64)
        unet_downsample(128, 4, init=init),                                         # (batch_size, 16, 16,  128)
        unet_downsample(256, 4, init=init),                                         # (batch_size,  8,  8,  256)
        unet_downsample(512, 4, init=init),                                         # (batch_size,  4,  4,  512)
        unet_downsample(512, 4, init=init),                                         # (batch_size,  2,  2,  512)
        unet_downsample(512, 4, init=init),                                         # (batch_size,  1,  1,  512)
    ]

    up_stack = [
        unet_upsample(512, 4, apply_dropout=True, init=init),                       # (batch_size,  2,  2, 1024)
        unet_upsample(512, 4, apply_dropout=True, init=init),                       # (batch_size,  4,  4, 1024)
        unet_upsample(256, 4, apply_dropout=True, init=init),                       # (batch_size,  8,  8,  512)
        unet_upsample(128, 4, init=init),                                           # (batch_size, 16, 16,  256)
        unet_upsample( 64, 4, init=init),                                           # (batch_size, 32, 32,  128)
        unet_upsample( 32, 4, init=init),                                           # (batch_size, 64, 64,   36)
    ]

    last = layers.Conv2D(output_channels, 4,                                        # (batch_size, 64, 64, 3/4 or max_palette_size)
                         padding="same",
                         kernel_initializer=init,
                         activation=last_activation)

    x = inputs

    # executing down sampling and saving the skip-connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # ignores the last skip and reverses them
    skips = list(reversed(skips[:-1]))

    # up samples and concatenates the partial results from skips
    for up, skip in zip(up_stack, [*skips, inputs]):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    if palette_quantization:
        input_palette = layers.Input(shape=[None, output_channels], name="input_palette")
        inputs = [inputs, input_palette]

        quantization_layer = keras_utils.DifferentiablePaletteQuantization(temperature)
        x = quantization_layer([x, input_palette])
        model = tf.keras.Model(inputs=inputs, outputs=x, name="unet-gen-quantized")
        model.quantization = quantization_layer
    else:
        model = tf.keras.Model(inputs=inputs, outputs=x, name="unet-gen")
    return model
