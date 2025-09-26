import os

import tensorflow as tf

import io_utils


# Some images have transparent pixels with colors other than black
# This function turns all transparent pixels to black
# TFJS does this by default, but TF does not
# The TFJS imported model was having bad inference because of this
def blacken_transparent_pixels(image):
    mask = tf.math.equal(image[..., 3], 0)
    repeated_mask = tf.repeat(mask, 4)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        image * 0.,
        image * 1.)
    return image


# This is used to convert RGBA images to RGB images
# replaces the alpha channel with a white color (only 100% transparent pixels)
def replace_alpha_with_white(image):
    mask = tf.math.equal(image[:, :, 3], 0)
    repeated_mask = tf.repeat(mask, 4)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        255.,
        image)

    # drops the A in RGBA
    image = image[:, :, :3]
    return image

def replace_alpha_with_black(image):
    mask = tf.math.equal(image[:, :, 3], 0)
    repeated_mask = tf.repeat(mask, 4)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        0.,
        image)

    # drops the A in RGBA
    image = image[:, :, :3]
    return image

def normalize(image):
    """
    Turns an image from the [0, 255] range into [-1, 1], keeping the same data type.
    Parameters
    ----------
    image a tensor representing an image
    Returns the image in the [-1, 1] range
    -------
    """
    return (image / 127.5) - 1


def denormalize(image):
    """
    Turns an image from the [-1, 1] range into [0, 255], keeping the same data type.
    Parameters
    ----------
    image a tensor representing an image
    Returns the image in the [0, 255] range
    -------
    """
    return (image + 1) * 127.5


# loads an image from the file system and transforms it for the network:
# (a) casts to float, (b) ensures transparent pixels are black-transparent, and (c)
# puts the values in the range of [-1, 1]
def load_image(path, should_normalize=True, size=64, input_channels=4, output_channels=4):
    image = None
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=input_channels)
        image = tf.reshape(image, (size, size, input_channels))
        image = tf.cast(image, "float32")
        if input_channels == 4:
            image = blacken_transparent_pixels(image)
        if output_channels == 3:
            image = replace_alpha_with_black(image)
        if should_normalize:
            image = normalize(image)
    except UnicodeDecodeError:
        print("Error opening image in ", path)
    return image


def augment_hue_rotation(image, seed, channels):
    if channels == 3:
        image = tf.image.stateless_random_hue(image, 0.5, seed)
    else:
        image_rgb, image_alpha = image[..., 0:3], image[..., 3]
        image_rgb = tf.image.stateless_random_hue(image_rgb, 0.5, seed)
        image = tf.concat([image_rgb, image_alpha[..., tf.newaxis]], axis=-1)
    return image


def augment_translation(images):
    image = tf.concat([*images], axis=-1)
    translate = tf.keras.layers.RandomTranslation(
        (-0.15, 0.075), 0.125, fill_mode="constant", interpolation="nearest")
    image = translate(image, training=True)
    images = tf.split(image, len(images), axis=-1)
    return tf.tuple(images)


def augment_two(first, second, should_rotate_hue, should_translate, channels):
    # hue rotation
    if should_rotate_hue:
        hue_seed = tf.random.uniform(shape=[2], minval=0, maxval=65536, dtype="int32")
        first = augment_hue_rotation(first, hue_seed, channels)
        second = augment_hue_rotation(second, hue_seed, channels)

    # translation
    if should_translate:
        first, second = augment_translation((first, second))

    return first, second


def normalize_two(first, second):
    return normalize(first), normalize(second)


def create_augmentation_with_prob(prob=0.8, should_augment_hue=True, should_augment_translation=True, channels=4):
    prob = tf.constant(prob)

    def augmentation_wrapper(first, second):
        choice = tf.random.uniform(shape=[])
        inside_augmentation_probability = choice < prob
        if inside_augmentation_probability:
            return augment_two(first, second, should_augment_hue, should_augment_translation, channels)
        else:
            return first, second

    return augmentation_wrapper


def create_indexed_image_loader(config, train_or_test_folder):
    """
    Returns a function which takes an integer in the range of [0, DATASET_SIZE-1] and loads some image file
    from the corresponding dataset (using image_number and DATASET_SIZES to decide) representing images by
    its palette and indexed colors.
    """
    domain_folders = config.domain_folders
    data_folders = config.data_folders
    dataset_sizes = config.train_sizes if train_or_test_folder == "train" else config.test_sizes
    output_channels = config.output_channels
    sprite_side_source = config.source_index
    sprite_side_target = config.target_index
    palette_ordering = config.palette_ordering
    max_palette_size = config.max_palette_size

    def load_indexed_images(dataset, image_number):
        folders = domain_folders
        source_path = tf.strings.join(
            [dataset, train_or_test_folder, folders[sprite_side_source], image_number + ".png"], os.sep)
        target_path = tf.strings.join(
            [dataset, train_or_test_folder, folders[sprite_side_target], image_number + ".png"], os.sep)

        source_image = tf.cast(load_image(
            source_path, should_normalize=False, input_channels=4, output_channels=output_channels), "int32")
        target_image = tf.cast(load_image(
            target_path, should_normalize=False, input_channels=4, output_channels=output_channels), "int32")

        # concatenates source and target so the colors in one have the same palette indices as the other
        concatenated_image = tf.concat([source_image, target_image], axis=-1)

        # finds the unique colors in both images
        palette = io_utils.extract_palette(concatenated_image, palette_ordering, max_palette_size, 4)

        # converts source and target_images from RGB into indexed, using the extracted palette
        source_image = io_utils.rgba_to_indexed(source_image, palette)
        target_image = io_utils.rgba_to_indexed(target_image, palette)

        return source_image, target_image, palette

    def load_images(image_number):
        image_number = tf.cast(image_number, "int32")

        # finds the dataset index and image number considering the param is an int
        # in an imaginary concatenated array of all datasets
        dataset_index = tf.constant(0, dtype="int32")

        def condition(which_image, which_dataset): return which_image >= tf.gather(
            dataset_sizes, which_dataset)
        def body(which_image, which_dataset): return [which_image - tf.gather(dataset_sizes, which_dataset),
                                                      which_dataset + 1]
        image_number, dataset_index = tf.while_loop(
            condition, body, [image_number, dataset_index])

        # gets the string pointing to the correct images
        dataset = tf.gather(data_folders, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # loads and transforms the images according to how the generator and discriminator expect them to be
        source_image, target_image, palette = load_indexed_images(
            dataset, image_number)
        return source_image, target_image, palette

    return load_images


def create_rgba_image_loader(config, train_or_test_folder):
    """
    Returns a function which takes an integer in the range of [0, DATASET_SIZE-1] and loads some image file
    from the corresponding dataset (using image_number and DATASET_SIZES to decide).
    """

    domain_folders = config.domain_folders
    data_folders = config.data_folders
    dataset_sizes = config.train_sizes if train_or_test_folder == "train" else config.test_sizes
    input_channels = config.input_channels
    output_channels = config.output_channels
    sprite_side_source = config.source_index
    sprite_side_target = config.target_index

    def load_images(image_number):
        image_number = tf.cast(image_number, "int32")

        # finds the dataset index and image number considering the param is an int
        # in an imaginary concatenated array of all datasets
        dataset_index = tf.constant(0, dtype="int32")

        def condition(which_image, which_dataset): return which_image >= tf.gather(
            dataset_sizes, which_dataset)
        def body(which_image, which_dataset): return [which_image - tf.gather(dataset_sizes, which_dataset),
                                                      which_dataset + 1]
        image_number, dataset_index = tf.while_loop(
            condition, body, [image_number, dataset_index])

        # gets the string pointing to the correct images
        dataset = tf.gather(data_folders, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # loads and transforms the images according to how the generator and discriminator expect them to be
        source_image = load_image(tf.strings.join(
            [dataset, os.sep, train_or_test_folder, os.sep, domain_folders[sprite_side_source], os.sep, image_number,
             ".png"]), False, input_channels=input_channels, output_channels=output_channels)
        target_image = load_image(tf.strings.join(
            [dataset, os.sep, train_or_test_folder, os.sep, domain_folders[sprite_side_target], os.sep, image_number,
             ".png"]), False, input_channels=input_channels, output_channels=output_channels)

        return source_image, target_image

    return load_images


def load_rgba_ds(config):
    train_size = config.train_size
    test_size = config.test_size
    batch_size = config.batch
    inner_channels = config.inner_channels
    prevent_augmentation = config.model == "baseline-no-aug"
    should_augment_hue = not prevent_augmentation and not config.no_hue
    should_augment_translation = not prevent_augmentation and not config.no_tran

    train_dataset = tf.data.Dataset.range(train_size).shuffle(train_size)
    test_dataset = tf.data.Dataset.range(test_size)

    train_dataset = train_dataset \
        .map(create_rgba_image_loader(config, "train"),
             num_parallel_calls=tf.data.AUTOTUNE)

    should_augment = should_augment_hue or should_augment_translation
    if should_augment:
        train_dataset = train_dataset \
            .map(create_augmentation_with_prob(0.8, should_augment_hue, should_augment_translation,
                                               inner_channels),
                 num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset \
        .map(normalize_two, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size)

    test_dataset = test_dataset.map(create_rgba_image_loader(config, "test"),
                                    num_parallel_calls=tf.data.AUTOTUNE) \
        .map(normalize_two, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size)
    return train_dataset, test_dataset


def load_indexed_ds(config):
    train_size = config.train_size
    test_size = config.test_size
    batch_size = config.batch

    train_dataset = tf.data.Dataset.range(train_size).shuffle(train_size)
    test_dataset = tf.data.Dataset.range(test_size)

    train_dataset = train_dataset \
        .map(create_indexed_image_loader(config, "train"),
             num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size)

    test_dataset = test_dataset \
        .map(create_indexed_image_loader(config, "test"),
             num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size)

    return train_dataset, test_dataset
