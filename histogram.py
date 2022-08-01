import tensorflow as tf


@tf.function
def calculate_component_histogram(component, projection1, projection2, color_intensities,
                                  histogram_domain, method, sigma_sqr, epsilon):
    # component         (batch, HW)
    # projection1       (batch, HW)
    # projection2       (batch, HW)
    # color_intensities (batch, HW, 1)
    # histogram_domain  (1, size), but it can broadcast to (batch, 1, size)

    Iu = tf.math.log(component + epsilon) - tf.math.log(projection1 + epsilon)      # (batch, HW)
    Iu = tf.expand_dims(Iu, -1)                                                     # (batch, HW, 1)

    Iv = tf.math.log(component + epsilon) - tf.math.log(projection2 + epsilon)      # (batch, HW)
    Iv = tf.expand_dims(Iv, -1)                                                     # (batch, HW, 1)

    # (batch, HW, 1) - (batch, 1, size) == (batch, HW, size)
    diff_u = tf.pow(Iu - histogram_domain, 2.) / sigma_sqr                          # (batch, HW, size)
    diff_v = tf.pow(Iv - histogram_domain, 2.) / sigma_sqr                          # (batch, HW, size)
    if method == "RBF":
        diff_u = tf.exp(-diff_u)  # radial basis function
        diff_v = tf.exp(-diff_v)
    elif method == "inverse-quadratic":
        diff_u = 1. / (1. + diff_u)
        diff_v = 1. / (1. + diff_v)

    a = tf.transpose(color_intensities * diff_u, [0, 2, 1])  # [(batch, HW, 1) * (batch, HW, size)]T = (batch, size, HW)
    component_histogram = tf.matmul(a, diff_v)  # (batch, size, size)

    return component_histogram


@tf.function
def calculate_rgbuv_histogram(image_batch, size=64, method="inverse-quadratic",
                              sigma=0.02, intensity_scale=True, hist_boundary=None):
    """
    Computes the color histogram of an image in a differentiable way.
    Adapted from HistoGAN:
    https://colab.research.google.com/drive/1dAF1_oAQ1c8OMLqlYA5V878pmpcnQ6_9?usp=sharing#scrollTo=mowAqNeraJij

    Parameters
    ----------
    image the image for which to compute the histogram
    size the square 2D size of the generated histogram. Default is 64
    method either "thresholding" (not differentiable), "RBF", or "inverse-quadratic". Default is "inverse-quadratic"
    sigma the sigma parameter of the kernel function. Default is 0.02
    intensity_scale boolean indicating whether to use the intensity scale. Default is True
    hist_boundary a list of histogram boundary values. Default is [-3, 3]

    Returns an image of the histogram
    -------

    """
    epsilon = 1e-6
    sigma_sqr = tf.pow(sigma, 2)
    histogram_domain = tf.expand_dims(tf.linspace(-3., 3., num=size), 0)  # (1, size)

    # we expect the image in [-1, 1], but the reference impl. needs it [0,1]
    image_batch = image_batch * 0.5 + 0.5

    batch_size = tf.shape(image_batch)[0]
    image_batch = image_batch[:, :, :, :3]

    # reshapes the image into I so it is a flat list of colors (per image in the batch)
    I = tf.reshape(image_batch, [batch_size, -1, 3])                                # (batch, H*W, 3)
    II = tf.pow(I, 2)                                                               # (batch, HW, 3)
    Iy = tf.sqrt(II[..., 0] + II[..., 1] + II[..., 2] + epsilon)[..., tf.newaxis]   # (batch, HW, 1)

    # separates the channels so the log-chroma coordinates u and v can be computed for R, G and B
    red, green, blue = I[..., 0], I[..., 1], I[..., 2]                              # (batch, HW)

    # each is                                                                         (batch, size, size)
    histogram_r = calculate_component_histogram(red, green, blue, Iy, histogram_domain, method, sigma_sqr, epsilon)
    histogram_g = calculate_component_histogram(green, red, blue, Iy, histogram_domain, method, sigma_sqr, epsilon)
    histogram_b = calculate_component_histogram(blue, red, green, Iy, histogram_domain, method, sigma_sqr, epsilon)
    histograms = tf.stack([histogram_r, histogram_g, histogram_b], -1)

    # normalization
    denominator = tf.reduce_sum(histograms, axis=[1, 2, 3], keepdims=True)
    histograms_normalized = histograms / denominator

    return histograms_normalized


@tf.function
def calculate_rgbuv_histogram_original(image_batch, size=64, method="inverse-quadratic",
                              sigma=0.02, intensity_scale=True, hist_boundary=None):
    # min = tf.reduce_min(image_batch)
    # max = tf.reduce_max(image_batch)
    # tf.cond min < -1 or max > 1:
    #     raise ValueError(f"image_batch passed to calculate_rgbuv_histogram had values outside [-1, 1]. "
    #                      f"Min {min} and max {max}")
    """
    Computes the color histogram of an image in a differentiable way.
    Adapted from HistoGAN:
    https://colab.research.google.com/drive/1dAF1_oAQ1c8OMLqlYA5V878pmpcnQ6_9?usp=sharing#scrollTo=mowAqNeraJij

    Parameters
    ----------
    image the image for which to compute the histogram
    size the square 2D size of the generated histogram. Default is 64
    method either "thresholding" (not differentiable), "RBF", or "inverse-quadratic". Default is "inverse-quadratic"
    sigma the sigma parameter of the kernel function. Default is 0.02
    intensity_scale boolean indicating whether to use the intensity scale. Default is True
    hist_boundary a list of histogram boundary values. Default is [-3, 3]

    Returns an image of the histogram
    -------

    """
    epsilon = 1e-6

    if hist_boundary is None:
        hist_boundary = tf.constant([-3., 3.], dtype="float32")

    # we expect the image in [-1, 1], but the reference impl. needs it [0,1]
    image_batch = image_batch * 0.5 + 0.5

    batch_size = tf.shape(image_batch)[0]
    channels = tf.shape(image_batch)[1]
    if channels > 3:
        image_batch = image_batch[:, :, :, :3]

    histograms = tf.zeros([batch_size, size, size, 3])
    # histograms = tf.unstack(histograms)
    index_in_batch = tf.constant(0, "int32")
    for image in image_batch:
        # flattens the image
        I = tf.reshape(image, (-1, 3))  # (H*W, 3)
        II = tf.pow(I, 2)  # (H*W, 3)

        if intensity_scale:
            Iy = tf.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + epsilon)  # (H*W)
            Iy = tf.expand_dims(Iy, -1)  # (H*W, 1)
        else:
            Iy = 1.

        Iu0 = tf.math.log(I[:, 0] + epsilon) - tf.math.log(I[:, 1] + epsilon)  # (H*W)
        Iu0 = tf.expand_dims(Iu0, -1)  # (H*W, 1)

        Iv0 = tf.math.log(I[:, 0] + epsilon) - tf.math.log(I[:, 2] + epsilon)  # (H*W)
        Iv0 = tf.expand_dims(Iv0, -1)  # (H*W, 1)

        histogram_domain = tf.expand_dims(tf.linspace(hist_boundary[0], hist_boundary[1], num=size), axis=0)  # (1, size)
        diff_u0 = tf.abs(Iu0 - histogram_domain)  # (H*W, size)
        diff_v0 = tf.abs(Iv0 - histogram_domain)

        reshaped_diff_u0 = tf.reshape(diff_u0, [-1, size])  # (H*W, size) -> (4096, 64)
        reshaped_diff_v0 = tf.reshape(diff_v0, [-1, size])
        if method == "thresholding":
            diff_u0 = reshaped_diff_u0 <= epsilon / 2.
            diff_v0 = reshaped_diff_v0 <= epsilon / 2.
        elif method == "RBF":
            diff_u0 = tf.pow(reshaped_diff_u0, 2.) / (sigma ** 2.)
            diff_v0 = tf.pow(reshaped_diff_v0, 2.) / (sigma ** 2.)
            diff_u0 = tf.exp(-diff_u0)  # radial basis function
            diff_v0 = tf.exp(-diff_v0)
        elif method == "inverse-quadratic":
            diff_u0 = tf.pow(reshaped_diff_u0, 2.) / (sigma ** 2.)
            diff_v0 = tf.pow(reshaped_diff_v0, 2.) / (sigma ** 2.)
            diff_u0 = 1. / (1. + diff_u0)
            diff_v0 = 1. / (1. + diff_v0)

        diff_u0 = tf.cast(diff_u0, "float32")  # (HW, size)
        diff_v0 = tf.cast(diff_v0, "float32")

        a = tf.transpose(Iy * diff_u0)  # [(HW, 1) * (HW, size)]T = (size, HW)
        histogram_r = tf.matmul(a, diff_v0)


        Iu1 = tf.math.log(I[:, 1] + epsilon) - tf.math.log(I[:, 0] + epsilon)  # (H*W)
        Iu1 = tf.expand_dims(Iu1, -1)  # (H*W, 1)

        Iv1 = tf.math.log(I[:, 1] + epsilon) - tf.math.log(I[:, 2] + epsilon)  # (H*W)
        Iv1 = tf.expand_dims(Iv1, -1)  # (H*W, 1)

        diff_u1 = tf.abs(Iu1 - histogram_domain)  # (H*W, size)
        diff_v1 = tf.abs(Iv1 - histogram_domain)

        reshaped_diff_u1 = tf.reshape(diff_u1, [-1, size])  # (H*W, size) -> (4096, 64)
        reshaped_diff_v1 = tf.reshape(diff_v1, [-1, size])
        if method == "thresholding":
            diff_u1 = reshaped_diff_u1  <= epsilon / 2.
            diff_v1 = reshaped_diff_v1  <= epsilon / 2.
        elif method == "RBF":
            diff_u1 = tf.pow(reshaped_diff_u1, 2.) / (sigma ** 2.)
            diff_v1 = tf.pow(reshaped_diff_v1, 2.) / (sigma ** 2.)
            diff_u1 = tf.exp(-diff_u1)  # radial basis function
            diff_v1 = tf.exp(-diff_v1)
        elif method == "inverse-quadratic":
            diff_u1 = tf.pow(reshaped_diff_u1, 2.) / (sigma ** 2.)
            diff_v1 = tf.pow(reshaped_diff_v1, 2.) / (sigma ** 2.)
            diff_u1 = 1. / (1. + diff_u1)
            diff_v1 = 1. / (1. + diff_v1)
        diff_u1 = tf.cast(diff_u1, "float32")
        diff_v1 = tf.cast(diff_v1, "float32")
        a = tf.transpose(Iy * diff_u1)
        histogram_g = tf.matmul(a, diff_v1)

        # Iu2 and Iv2
        Iu2 = tf.math.log(I[:, 2] + epsilon) - tf.math.log(I[:, 0] + epsilon)  # (H*W)
        Iu2 = tf.expand_dims(Iu2, -1)  # (H*W, 1)

        Iv2 = tf.math.log(I[:, 2] + epsilon) - tf.math.log(I[:, 1] + epsilon)  # (H*W)
        Iv2 = tf.expand_dims(Iv2, -1)  # (H*W, 1)

        diff_u2 = tf.abs(Iu2 - histogram_domain)  # (H*W, size)
        diff_v2 = tf.abs(Iv2 - histogram_domain)

        reshaped_diff_u2 = tf.reshape(diff_u2, [-1, size])  # (H*W, size) -> (4096, 64)
        reshaped_diff_v2 = tf.reshape(diff_v2, [-1, size])
        if method == "thresholding":
            diff_u2 = reshaped_diff_u2  <= epsilon / 2.
            diff_v2 = reshaped_diff_v2  <= epsilon / 2.
        elif method == "RBF":
            diff_u2 = tf.pow(reshaped_diff_u2, 2.) / (sigma ** 2.)
            diff_v2 = tf.pow(reshaped_diff_v2, 2.) / (sigma ** 2.)
            diff_u2 = tf.exp(-diff_u2)  # radial basis function
            diff_v2 = tf.exp(-diff_v2)
        elif method == "inverse-quadratic":
            diff_u2 = tf.pow(reshaped_diff_u2, 2.) / (sigma ** 2.)
            diff_v2 = tf.pow(reshaped_diff_v2, 2.) / (sigma ** 2.)
            diff_u2 = 1. / (1. + diff_u2)
            diff_v2 = 1. / (1. + diff_v2)
        diff_u2 = tf.cast(diff_u2, "float32")
        diff_v2 = tf.cast(diff_v2, "float32")
        a = tf.transpose(Iy * diff_u2)
        histogram_b = tf.matmul(a, diff_v2)
        histograms = tf.tensor_scatter_nd_update(histograms, tf.expand_dims([index_in_batch], -1), tf.expand_dims(tf.stack([histogram_r, histogram_g, histogram_b], axis=-1), 0))

        index_in_batch += 1

    # reassembles histograms from python list to tensor
    histograms = tf.stack(histograms, axis=0)
    if tf.rank(histograms) == 3:
        histograms = tf.expand_dims(histograms, -1)

    # normalization
    denominator = tf.reduce_sum(histograms, axis=[1, 2, 3], keepdims=True)
    histograms_normalized = histograms / denominator

    return histograms_normalized


def hellinger_loss(y_true, y_pred):
    batch_size = tf.cast(tf.shape(y_true)[0], "float32")
    # Hellinger distance between the two histograms:
    # 1/sqrt(2) * ||sqrt(H_true) - sqrt(H_pred)||
    return (1. / tf.sqrt(2.) * tf.sqrt(
        tf.reduce_sum(tf.pow(tf.sqrt(y_pred) - tf.sqrt(y_true), 2.)))) / batch_size


def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def l2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.pow(y_true - y_pred, 2))
