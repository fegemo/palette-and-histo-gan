import tensorflow as tf
from tensorflow.keras import layers
from configuration import *
from tensorflow_addons import layers as tfalayers


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
        result.add(tfalayers.InstanceNormalization())
    result.add(layers.LeakyReLU())

    return result


def unet_upsample(filters, size, apply_dropout=False, init=tf.random_normal_initializer(0., 0.02)):
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2, padding="same", kernel_initializer=init, use_bias=False))

    result.add(tfalayers.InstanceNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    
    return result


def PatchDiscriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="target_image")

    x = layers.concatenate([target_image, source_image])  # (batch_size, 64, 64, channels*2)

    down = unet_downsample(64, 4, False)(x)  # (batch_size, 32, 32, 64)
    last = layers.Conv2D(1, 4, padding="same",
                            kernel_initializer=initializer)(down)  # (batch_size, 32, 32, 1)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=last, name="patch-disc")


def IndexedPatchDiscriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1], name="target_image")

    x = layers.concatenate([target_image, source_image])  # (batch_size, 64, 64, 2)

    down = unet_downsample(64, 4, False)(x)  # (batch_size, 32, 32, 64)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(down)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=last, name="indexed-patch-disc")


def UnetGenerator():
    init = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS]) #(batch_size, 64, 64, 4)

    down_stack = [
        unet_downsample( 64, 4, apply_batchnorm=False, init=init),  # (batch_size, 32, 32,   64)
        unet_downsample(128, 4, init=init),                         # (batch_size, 16, 16,  128)
        unet_downsample(256, 4, init=init),                         # (batch_size,  8,  8,  256)
        unet_downsample(512, 4, init=init),                         # (batch_size,  4,  4,  512)
        unet_downsample(512, 4, init=init),                         # (batch_size,  2,  2,  512)
        unet_downsample(512, 4, init=init),                         # (batch_size,  1,  1,  512)
    ]

    up_stack = [
        unet_upsample(512, 4, apply_dropout=True, init=init),       # (batch_size,  2,  2, 1024)
        unet_upsample(512, 4, apply_dropout=True, init=init),       # (batch_size,  4,  4, 1024)
        unet_upsample(256, 4, apply_dropout=True, init=init),       # (batch_size,  8,  8,  512)
        unet_upsample(128, 4, init=init),                           # (batch_size, 16, 16,  256)
        unet_upsample( 64, 4, init=init),                           # (batch_size, 32, 32,  128)
        unet_upsample( 32, 4, init=init),                           # (batch_size, 64, 64,   36)
    ]

    last = layers.Conv2D(OUTPUT_CHANNELS, 4,
                                     strides=1,
                                     padding="same",
                                     kernel_initializer=init,
                                     activation="tanh")  # (batch_size, 64, 64, 4)

    x = inputs

    # downsampling e adicionando as skip-connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # ignora a última skip e inverte a ordem
    skips = list(reversed(skips[:-1]))

    # camadas de upsampling e skip connections
    for up, skip in zip(up_stack, [*skips, inputs]):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="unet-gen")


def IndexedUnetGenerator():
    init = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1], name="input_image") #(batch_size, 64, 64, 1)

    down_stack = [
        unet_downsample( 32, 4, apply_batchnorm=False, init=init),  # (batch_size, 32, 32,   64)
        unet_downsample( 64, 4, init=init),                         # (batch_size, 16, 16,  128)
        unet_downsample(128, 4, init=init),                         # (batch_size,  8,  8,  256)
        unet_downsample(256, 4, init=init),                         # (batch_size,  4,  4,  512)
        unet_downsample(512, 4, init=init),                         # (batch_size,  2,  2,  512)
        unet_downsample(512, 4, init=init),                         # (batch_size,  1,  1,  512)
    ]

    up_stack = [
        unet_upsample(512, 4, apply_dropout=True, init=init),       # (batch_size,  2,  2, 1024)
        unet_upsample(256, 4, apply_dropout=True, init=init),       # (batch_size,  4,  4, 1024)
        unet_upsample(128, 4, apply_dropout=True, init=init),       # (batch_size,  8,  8,  512)
        unet_upsample( 64, 4, init=init),       # (batch_size, 16, 16,  256)
        unet_upsample( 64, 4, init=init),       # (batch_size, 32, 32,  128)
        unet_upsample(128, 4, init=init),       # (batch_size, 64, 64,   64)
    ]

    last = layers.Conv2D(MAX_PALETTE_SIZE, 4,
                                     strides=1,
                                     padding="same",
                                     kernel_initializer=init,
                                     activation="softmax"
                         )  # (batch_size, 64, 64, 4)

    x = inputs

    # downsampling e adicionando as skip-connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # ignora a última skip e inverte a ordem
    skips = list(reversed(skips[:-1]))

    # camadas de upsampling e skip connections
    for up, skip in zip(up_stack, [*skips, inputs]):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

