import shutil
import io
import tensorflow as tf
from matplotlib import pyplot as plt

from configuration import *


def ensure_folder_structure(*folders):
    is_absolute_path = os.path.isabs(folders[0])
    provided_paths = []
    for path_part in folders:
        provided_paths.extend(path_part.split(os.sep))
    folder_path = os.getcwd() if not is_absolute_path else "/"

    for folder_name in provided_paths:
        folder_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)


def delete_folder(path):
    shutil.rmtree(path, ignore_errors=True)


@tf.function
def extract_palette(image, palette_ordering, channels=OUTPUT_CHANNELS):
    """
    Extracts the unique colors from an image (3D tensor)
    Parameters
    ----------
    image a 3D tensor with shape (height, width, channels)
    palette_ordering either "grayness", "top2bottom", "bottom2top", or "shuffled"
    channels the number of channels of the image

    Returns a tensor of colors (RGB) sorted by the number of times each one appears and from dark to light as a
    second sorting key.
    -------
    """
    # incoming image shape: (IMG_SIZE, IMG_SIZE, channels)
    # reshaping to: (IMG_SIZE*IMG_SIZE, channels)
    image = tf.cast(image, "int32")
    image = tf.reshape(image, [-1, channels])

    if palette_ordering == "top2bottom":
        # the UniqueWithCountsV2 sweeps the image from the top-left to the bottom-right corner
        colors, _, count = tf.raw_ops.UniqueWithCountsV2(x=image, axis=[0])
    elif palette_ordering == "bottom2top":
        image = image[::-1]  # sorting by appearance: bottom-right to top-left
        colors, _, count = tf.raw_ops.UniqueWithCountsV2(x=image, axis=[0])
    elif palette_ordering == "grayness":
        colors, _, count = tf.raw_ops.UniqueWithCountsV2(x=image, axis=[0])
        gray_coefficients = tf.constant([0.2989, 0.5870, 0.1140, 0.])[..., tf.newaxis]
        grayness = tf.squeeze(tf.matmul(tf.cast(colors, "float32"), gray_coefficients))
        indices_sorted_by_grayness = tf.argsort(grayness, direction="ASCENDING", stable=True)
        colors = tf.gather(colors, indices_sorted_by_grayness)
    else:  # shuffled
        colors, _, count = tf.raw_ops.UniqueWithCountsV2(x=image, axis=[0])
        colors = tf.random.shuffle(colors)

    # fills the palette to have 256 colors, so batches can be created (otherwise they can't, tf complains)
    num_colors = tf.shape(colors)[0]
    fillers = tf.repeat([INVALID_INDEX_COLOR], [MAX_PALETTE_SIZE - num_colors], axis=0)
    colors = tf.concat([colors, fillers], axis=0)

    return colors


@tf.function
def rgba_to_single_int(values_in_rgba):
    shape = tf.shape(values_in_rgba)
    converted = tf.zeros(shape=shape[:-1], dtype="int32")
    # multiplier: 2 ** [24, or 16, or 8, or 0]
    for i, multiplier in enumerate([16777216, 65536, 256, 0]):
        converted += values_in_rgba[..., i] * multiplier
    return converted


@tf.function
def rgba_to_indexed(image, palette):
    original_shape = tf.shape(image)
    flattened_image = tf.reshape(image, [-1, original_shape[-1]])
    num_pixels, num_components = tf.shape(flattened_image)[0], tf.shape(flattened_image)[1]

    indices = flattened_image == palette[:, None]
    row_sums = tf.reduce_sum(tf.cast(indices, "int32"), axis=2)
    results = tf.cast(tf.where(row_sums == num_components), "int32")

    color_indices, pixel_indices = results[:, 0], results[:, 1]
    pixel_indices = tf.expand_dims(pixel_indices, -1)

    indexed = tf.scatter_nd(pixel_indices, color_indices, [num_pixels])
    indexed = tf.reshape(indexed, [original_shape[0], original_shape[1], 1])
    return indexed


@tf.function
def indexed_to_rgba(indexed_image, palette):
    image_shape = tf.shape(indexed_image)
    image_rgb = tf.gather(palette, indexed_image)

    # now the shape is (HEIGHT, WIDTH, 1, CHANNELS), so we need to reshape
    image_rgb = tf.reshape(image_rgb, [image_shape[0], image_shape[1], -1])
    return image_rgb


def plot_to_image(matplotlib_figure, channels=OUTPUT_CHANNELS):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", transparent=True)
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(matplotlib_figure)
    buffer.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buffer.getvalue(), channels=channels)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
    

def seconds_to_human_readable(time):
    days = time // 86400         # (60 * 60 * 24)
    hours = time // 3600 % 24    # (60 * 60) % 24
    minutes = time // 60 % 60 
    seconds = time % 60
    
    time_string = ""
    if days > 0:
        time_string += f"{days:.0f} day{'s' if days > 1 else ''}, "
    if hours > 0 or days > 0:
        time_string += f"{hours:02.0f}h:"
    time_string += f"{minutes:02.0f}m:{seconds:02.0f}s"
    
    return time_string
