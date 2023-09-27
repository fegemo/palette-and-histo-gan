import random

import tensorflow as tf
from tqdm import tqdm

import histogram
from dataset_utils import blacken_transparent_pixels, normalize
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import frechet_inception_distance as fid
from networks import UnetGenerator
from pix2pix_model import PostProcessGenerator


def load_image(path):
    image = None
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image)
        image = tf.reshape(image, (64, 64, 4))
        image = tf.cast(image, "float32")
        image = blacken_transparent_pixels(image)
        image = normalize(image)
    finally:
        return image


"""
Script that loads a model and then generates an image per sample comparing the post-processing
using RGB, YUV, and CIELAB representations.
"""
if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(47)
    random.seed(47)

    # loads the model
    model = UnetGenerator(4, 4, "tanh")
    surrogate_root = tf.train.Checkpoint(generator=model)
    surrogate_root.restore(tf.train.latest_checkpoint(f"output-gmod/new-post-process/front-to-right/baseline/20230911-103923/training-checkpoints"))
    # surrogate_root.restore(tf.train.latest_checkpoint(f"output-gmod/postprocess/none,no-aug/front-to-right/baseline/20230819-102720/training-checkpoints"))
    # model = tf.keras.models.load_model(f"models/py/generator/front-to-right/baseline")

    post_processors = [PostProcessGenerator(model, representation) for representation in ["rgb", "yuv", "cielab"]]

    # specify the desired images
    batch_size = 4
    random_list = list(range(44))
    random.shuffle(random_list)

    path_creator = lambda domain, n: f"datasets/rpg-maker-xp/test/{domain}/{n}.png"
    test_real_images = []
    test_fake_images = []
    test_rgb__images = []
    test_yuv__images = []
    test_lab__images = []

    # starting to generate the images
    print("Starting to generate the images...")
    for b in tqdm(range(44 // 4), desc=f"Batch of {batch_size} images", position=0):
        selected_image_indices = random_list[b * batch_size: (b + 1) * batch_size]

        # specify the paths to the folders
        image_paths = [(path_creator("2-front", n), path_creator("3-right", n)) for n in selected_image_indices]

        # loads the images from the batch
        examples = [tuple([load_image(path[0]), load_image(path[1])]) for path in image_paths]
        examples = [example for example in examples if not None in example]

        # generates the images from this batch
        titles = ["Input", "Target", "Generated", "Process. RGB", "Process. YUV", "Process. CIELAB"]
        num_cols = len(titles)
        num_rows = batch_size
        figure = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
        for i, (source, target) in enumerate(tqdm(examples, desc=f"From batch {b}", position=1, leave=False)):
            # for each image in the batch, generate, post-process, and calculate the histograms
            generated = model(source[tf.newaxis, ...], training=True)
            processed_images = [post_processor(source[tf.newaxis, ...]) for post_processor in post_processors]
            images = [source[tf.newaxis, ...], target[tf.newaxis, ...], generated] + processed_images
            images = tf.concat(images, axis=0)
            histograms = histogram.calculate_rgbuv_histogram(images)
            histograms = tf.image.central_crop(histograms, 0.7)

            for j, title in enumerate(titles):
                idx = i * num_cols + j + 1
                axes = plt.subplot(num_rows, num_cols, idx)
                plt.title(title if i == 0 else "", fontdict={"fontsize": 24})
                plt.imshow(images[j] * 0.5 + 0.5)
                histogram_pad = OffsetImage(tf.clip_by_value(histograms[j] * 100., 0., 1.), zoom=1.9)
                histogram_ab = AnnotationBbox(histogram_pad, (1, 0), xycoords="axes fraction", frameon=False, pad=0, box_alignment=(1, 0))
                axes.add_artist(histogram_ab)
                plt.axis("off")

            test_real_images.append(target)
            test_fake_images.append(generated[0])
            test_rgb__images.append(processed_images[0][0])
            test_yuv__images.append(processed_images[1][0])
            test_lab__images.append(processed_images[2][0])

        figure.tight_layout()
        plt.savefig(f"output-postprocessed/batch-{b}.png", transparent=True)
        plt.close(figure)

    print("Finished generating the images.")

    print("Starting the evaluations...")
    test_real_images = tf.stack(test_real_images)
    test_fake_images = tf.stack(test_fake_images)
    test_rgb__images = tf.stack(test_rgb__images, axis=0)
    test_yuv__images = tf.stack(test_yuv__images, axis=0)
    test_lab__images = tf.stack(test_lab__images, axis=0)

    print(f"Starting to evaluate L1...")
    l1_none = tf.reduce_mean(tf.abs(test_real_images - test_fake_images))
    l1_rgb_ = tf.reduce_mean(tf.abs(test_real_images - test_rgb__images))
    l1_yuv_ = tf.reduce_mean(tf.abs(test_real_images - test_yuv__images))
    l1_lab_ = tf.reduce_mean(tf.abs(test_real_images - test_lab__images))
    print(f"Finished to evaluate L1.")

    print(f"Starting to evaluate FID...")
    fid_none = fid.compare(test_real_images.numpy(), test_fake_images.numpy())
    fid_rgb_ = fid.compare(test_real_images.numpy(), test_rgb__images.numpy())
    fid_yuv_ = fid.compare(test_real_images.numpy(), test_yuv__images.numpy())
    fid_lab_ = fid.compare(test_real_images.numpy(), test_lab__images.numpy())
    print(f"Finished to evaluate FID.")

    print("\"Augmentation\";\"Best FID\";\"Best L1\";\"Best FID\";\"Best L1\";\"Best FID\";\"Best L1\";\"Best FID\";\"Best L1\"")
    print(f"XXX;{fid_none:.3f};{l1_none:.5f};{fid_rgb_:.3f};{l1_rgb_:.5f};{fid_yuv_:.3f};{l1_yuv_:.5f};{fid_lab_:.3f};{l1_lab_:.5f}")



