import tensorflow as tf


@tf.function
def calculate_component_histogram(component, projection1, projection2, color_intensities,
                                  histogram_domain, method, sigma_sqr, epsilon):
    # component         (batch, HW)
    # projection1       (batch, HW)
    # projection2       (batch, HW)
    # color_intensities (batch, HW, 1)
    # histogram_domain  (1, size), but it can broadcast to (batch, 1, size)

    Iu = tf.math.log(component + epsilon) - tf.math.log(projection1 + epsilon)  # (batch, HW)
    Iu = tf.expand_dims(Iu, -1)  # (batch, HW, 1)

    Iv = tf.math.log(component + epsilon) - tf.math.log(projection2 + epsilon)  # (batch, HW)
    Iv = tf.expand_dims(Iv, -1)  # (batch, HW, 1)

    # (batch, HW, 1) - (batch, 1, size) == (batch, HW, size)
    diff_u = tf.pow(Iu - histogram_domain, 2.) / sigma_sqr  # (batch, HW, size)
    diff_v = tf.pow(Iv - histogram_domain, 2.) / sigma_sqr  # (batch, HW, size)
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
def calculate_rgbuv_histogram(image_batch, size=64, method="inverse-quadratic", sigma=0.02):
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
    I = tf.reshape(image_batch, [batch_size, -1, 3])  # (batch, H*W, 3)
    II = tf.pow(I, 2)  # (batch, HW, 3)
    Iy = tf.sqrt(II[..., 0] + II[..., 1] + II[..., 2] + epsilon)[..., tf.newaxis]  # (batch, HW, 1)

    # separates the channels so the log-chroma coordinates u and v can be computed for R, G and B
    red, green, blue = I[..., 0], I[..., 1], I[..., 2]  # (batch, HW)

    # each is (batch, size, size)
    histogram_r = calculate_component_histogram(red, green, blue, Iy, histogram_domain, method, sigma_sqr, epsilon)
    histogram_g = calculate_component_histogram(green, red, blue, Iy, histogram_domain, method, sigma_sqr, epsilon)
    histogram_b = calculate_component_histogram(blue, red, green, Iy, histogram_domain, method, sigma_sqr, epsilon)
    histograms = tf.stack([histogram_r, histogram_g, histogram_b], -1)

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
