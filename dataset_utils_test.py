import tensorflow as tf

from dataset_utils import blacken_transparent_pixels


class TestBlackenTransparentPixels(tf.test.TestCase):
    def test_blacken_transparent_pixels(self):
        image_with_transparency = tf.constant([[
            [[1., 1., 1., 0.], [0., 0., 0., 0.]],
            [[.4, .4, .4, .1], [0., 0., 0., 1.]]
        ]])
        expected_blackened = tf.constant([[
            [[0., 0., 0., 0.], [0., 0., 0., 0.]],
            [[.4, .4, .4, .1], [0., 0., 0., 1.]]
        ]])

        result = blacken_transparent_pixels(image_with_transparency)
        self.assertAllEqual(result, expected_blackened)

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()


if __name__ == '__main__':
    tf.test.main()
