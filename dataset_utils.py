import tensorflow as tf

import io_utils
from configuration import *


# Some images have transparent pixels with colors other than black
# This function turns all transparent pixels to black
# TFJS does this by default, but TF does not
# The TFJS imported model was having bad inference because of this
def blacken_transparent_pixels(image):
    mask = tf.math.equal(image[:, :, 3], 0)
    repeated_mask = tf.repeat(mask, INPUT_CHANNELS)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        image * 0.,
        image * 1.)
    return image


# replaces the alpha channel with a white color (only 100% transparent pixels)
def replace_alpha_with_white(image):
    mask = tf.math.equal(image[:, :, 3], 0)
    repeated_mask = tf.repeat(mask, INPUT_CHANNELS)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        255.,
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
def load_image(path, should_normalize=True):
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=INPUT_CHANNELS)
        image = tf.reshape(image, (IMG_SIZE, IMG_SIZE, INPUT_CHANNELS))
        image = tf.cast(image, "float32")
        image = blacken_transparent_pixels(image)
        if should_normalize:
            image = normalize(image)
    except UnicodeDecodeError:
        print("Error opening image in ", path)
    return image


def augment_hue_rotation(image, seed):
    image_rgb, image_alpha = image[..., 0:3], image[..., 3]
    image_rgb = tf.image.stateless_random_hue(image_rgb, 0.5, seed)
    image = tf.concat([image_rgb, image_alpha[..., tf.newaxis]], axis=-1)
    return image


def augment_translation(images):
    image = tf.concat([*images], axis=-1)
    translate = tf.keras.layers.RandomTranslation(
        (-0.15, 0.075), 0.125, fill_mode="constant", interpolation="nearest")
    image = translate(image)
    images = tf.split(image, len(images), axis=-1)
    return tf.tuple(images)


def augment_two(first, second):
    # hue rotation
    hue_seed = tf.random.uniform(
        shape=[2], minval=0, maxval=65536, dtype="int32")
    first = augment_hue_rotation(first, hue_seed)
    second = augment_hue_rotation(second, hue_seed)
    # translation
    first, second = augment_translation((first, second))
    return first, second


def normalize_two(first, second):
    return normalize(first), normalize(second)


def create_augmentation_with_prob(prob=0.8):
    prob = tf.constant(prob)

    def augmentation_wrapper(first, second):
        choice = tf.random.uniform(shape=[])
        should_augment = choice < prob
        if should_augment:
            return augment_two(first, second)
        else:
            return first, second

    return augmentation_wrapper


def create_indexed_image_loader(sprite_side_source, sprite_side_target, dataset_sizes, train_or_test_folder,
                                palette_ordering):
    """
    Returns a function which takes an integer in the range of [0, DATASET_SIZE-1] and loads some image file
    from the corresponding dataset (using image_number and DATASET_SIZES to decide) representing images by
    its palette and indexed colors.
    """

    def load_indexed_images(dataset, image_number):
        folders = DIRECTION_FOLDERS
        source_path = tf.strings.join(
            [dataset, train_or_test_folder, folders[sprite_side_source], image_number + ".png"], os.sep)
        target_path = tf.strings.join(
            [dataset, train_or_test_folder, folders[sprite_side_target], image_number + ".png"], os.sep)

        source_image = tf.cast(load_image(
            source_path, should_normalize=False), "int32")
        target_image = tf.cast(load_image(
            target_path, should_normalize=False), "int32")

        # concatenates source and target so the colors in one have the same palette indices as the other
        concatenated_image = tf.concat([source_image, target_image], axis=-1)

        # finds the unique colors in both images
        palette = io_utils.extract_palette(
            concatenated_image, palette_ordering)

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
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # loads and transforms the images according to how the generator and discriminator expect them to be
        source_image, target_image, palette = load_indexed_images(
            dataset, image_number)
        return source_image, target_image, palette

    return load_images


def create_rgba_image_loader(sprite_side_source, sprite_side_target, dataset_sizes, train_or_test_folder):
    """
    Returns a function which takes an integer in the range of [0, DATASET_SIZE-1] and loads some image file
    from the corresponding dataset (using image_number and DATASET_SIZES to decide).
    """

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
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # loads and transforms the images according to how the generator and discriminator expect them to be
        input_image = load_image(tf.strings.join(
            [dataset, os.sep, train_or_test_folder, os.sep, DIRECTION_FOLDERS[sprite_side_source], os.sep, image_number,
             ".png"]), False)
        real_image = load_image(tf.strings.join(
            [dataset, os.sep, train_or_test_folder, os.sep, DIRECTION_FOLDERS[sprite_side_target], os.sep, image_number,
             ".png"]), False)

        return input_image, real_image

    return load_images


def load_rgba_ds(source_direction, target_direction, augment=True):
    train_dataset = tf.data.Dataset.range(TRAIN_SIZE).shuffle(TRAIN_SIZE)
    test_dataset = tf.data.Dataset.range(TEST_SIZE).shuffle(TEST_SIZE)

    train_dataset = train_dataset \
        .map(create_rgba_image_loader(source_direction, target_direction, TRAIN_SIZES, "train"),
             num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        train_dataset = train_dataset \
            .map(create_augmentation_with_prob(0.8), num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset \
        .map(normalize_two, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(BATCH_SIZE)

    test_dataset = test_dataset.map(create_rgba_image_loader(source_direction, target_direction, TEST_SIZES, "test"),
                                    num_parallel_calls=tf.data.AUTOTUNE) \
        .map(normalize_two, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(BATCH_SIZE)
    return train_dataset, test_dataset


def load_indexed_ds(source_direction, target_direction, palette_ordering):
    train_dataset = tf.data.Dataset.range(TRAIN_SIZE).shuffle(TRAIN_SIZE)
    test_dataset = tf.data.Dataset.range(TEST_SIZE).shuffle(TEST_SIZE)

    train_dataset = train_dataset \
        .map(create_indexed_image_loader(source_direction, target_direction, TRAIN_SIZES, "train", palette_ordering),
             num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(BATCH_SIZE)

    test_dataset = test_dataset \
        .map(create_indexed_image_loader(source_direction, target_direction, TEST_SIZES, "test", palette_ordering),
             num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(BATCH_SIZE)

    return train_dataset, test_dataset


def load_dataset(options):
    if options.model == "baseline-no-aug":
        train_ds, test_ds = load_rgba_ds(options.source_index, options.target_index, augment=False)
    elif options.model == "baseline":
        train_ds, test_ds = load_rgba_ds(options.source_index, options.target_index)
    elif options.model == "indexed":
        train_ds, test_ds = load_indexed_ds(options.source_index, options.target_index, palette_ordering=options.palette_ordering)
    elif options.model == "histogram":
        train_ds, test_ds = load_rgba_ds(options.source_index, options.target_index)
    else:
        raise SystemExit(
            f"The specified model {options.model} was not recognized.")

    return train_ds, test_ds
