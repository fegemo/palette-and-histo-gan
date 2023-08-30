import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

from histogram import calculate_rgbuv_histogram, hellinger_loss, l1_loss, l2_loss, hellinger_distance

# image_files = [f"histo-tests{os.sep}{f}" for f in os.listdir("histo-tests") if os.path.isfile(os.path.join("histo-tests", f))]
image_files = [f"histo-tests{os.sep}{f}" for f in ["postprocess.png"]]
horizontal_tiles = 4
vertical_tiles = 6
crop_size = 64
crops_per_image = 1 * vertical_tiles
number_of_total_crops = len(image_files) * crops_per_image

images_real = np.ndarray([number_of_total_crops, crop_size, crop_size, 4], np.float32)
images_fake = np.ndarray([number_of_total_crops, crop_size, crop_size, 4], np.float32)
images_ppcs = np.ndarray([number_of_total_crops, crop_size, crop_size, 4], np.float32)

crop_i = 0
for image_file in image_files:
    image = Image.open(image_file)
    image = image.resize((crop_size * horizontal_tiles, crop_size * vertical_tiles), Image.Resampling.NEAREST)
    pixdata = image.load()
    width, height = image.size
    for i in range(height):
        for j in range(width):
            if pixdata[j, i][3] == 0:
                pixdata[j, i] = (0, 0, 0, 0)


    for i in range(vertical_tiles):
        for j in range(horizontal_tiles):
            if j == 0:
                continue

            top = crop_size * i
            left = crop_size * j
            bottom = top + crop_size
            right = left + crop_size

            crop = image.crop((left, top, right, bottom))
            if j == 1:
                images = images_real
            elif j == 2:
                images =  images_fake
            else:
                images = images_ppcs
            images[crop_i] = np.array(crop)[..., :4] / 255.
            if j == 3:
                crop_i += 1


histograms_real = calculate_rgbuv_histogram(images_real*2.-1.).numpy()
histograms_fake = calculate_rgbuv_histogram(images_fake*2.-1.).numpy()
histograms_ppcs = calculate_rgbuv_histogram(images_ppcs*2.-1.).numpy()

hellinger_losses = hellinger_loss(histograms_real, histograms_fake)
l1_losses = l1_loss(histograms_real, histograms_fake)
l2_losses = l2_loss(histograms_real, histograms_fake)

for c in range(histograms_real.shape[0]):
    # if c == 0 or (c+1) % 6 != 0:
    #     continue
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.imshow(images_real[c])
    plt.axis("off")
    plt.subplot(3, 2, 2)
    plt.imshow(histograms_real[c]*200.)
    plt.axis("off")
    plt.subplot(3, 2, 3)
    plt.imshow(images_fake[c])
    plt.axis("off")
    plt.subplot(3, 2, 4)
    plt.imshow(histograms_fake[c]*200.)
    plt.axis("off")
    plt.subplot(3, 2, 5)
    plt.imshow(images_ppcs[c])
    plt.axis("off")
    plt.subplot(3, 2, 6)
    plt.imshow(histograms_ppcs[c]*200.)
    plt.axis("off")
    hdist = hellinger_distance(histograms_real[c], histograms_fake[c])
    l1dist = np.sum(np.abs(histograms_real[c] - histograms_fake[c]))
    l2dist = np.sum(np.power(histograms_real[c] - histograms_fake[c], 2))
    # plt.suptitle(f"hdist: {hdist:.5f} / hellinger: {hellinger_losses:.5f} / l1: {l1_losses:.5f}, l2: {l2_losses:.5f}")
    plt.suptitle(f"hdist: {hdist:.5f}  / l1: {l1dist:.5f}, l2: {l2dist:.5f}")

plt.show()
