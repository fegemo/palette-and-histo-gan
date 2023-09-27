import tensorflow as tf
from tqdm import tqdm

from dataset_utils import blacken_transparent_pixels, normalize
from matplotlib import pyplot as plt


# class that contains one model for each source/target direction from back, left, front, right
class AllSidesModel:
    def __init__(self):
        self.domains = ['back', 'left', 'front', 'right']
        self.models = {}
        print("Loading individual models...")
        for source_index, source in enumerate(tqdm(self.domains, desc=" Source domain", position=0)):
            self.models[source] = {}
            for target_index, target in enumerate(tqdm(self.domains, desc=" Target domain", position=1, leave=False)):
                self.models[source][target] = tf.keras.models.load_model(f"..\\..\\gmod-outputs\\all-sides-output\\{source}-to-{target}\\baseline\\models\\py\\generator")
                # self.models[source][target] = tf.keras.models.load_model(f"..\\..\\gmod-outputs\\all-sides-output-aiide\\models\\py\\generator\\{source}-to-{target}\\baseline")

    def generate(self, source, target, image):
        model = self.models[source][target]
        return model(image, training=True)

    def generate_all_to_all(self, source_images, save_name=None):
        titles = ['Source', 'To Back', 'To Left', 'To Front', 'To Right']
        num_cols = len(titles)
        num_rows = 4

        figure = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j + 1

                if j == 0:
                    generated_image = source_images[i]
                else:
                    source_domain = self.domains[i]
                    target_domain = self.domains[j - 1]
                    source_image = tf.expand_dims(source_images[i], 0)
                    generated_image = self.generate(source_domain, target_domain, source_image)[0]

                plt.subplot(num_rows, num_cols, idx)
                plt.title(titles[j] if i == 0 else "", fontdict={"fontsize": 24})
                plt.imshow(generated_image * 0.5 + 0.5)
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name, transparent=True)
        plt.close(figure)

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
Script that loads all single side to side models (with side E {back, left, front, right})
and generates an image, per example, with 5 columns and 4 rows. The first column is the 
source image on row and the others are the generated image for the column's particular
target side.
It was used for the SBGames'22 extended paper for Graphical Models (GMOD 2023).
"""
if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')

    # loads the model
    star_model = AllSidesModel()

    # specify the desired images
    batch_size = 20
    for b in tqdm(range(0, 93), desc=f"Batch of {batch_size} images", position=0):
        selected_image_indices = range(b * batch_size, (b + 1) * batch_size) #[783, 1770]

        # specify the paths to the folders
        domain_folders = ['0-back', '1-left', '2-front', '3-right']
        domain_folders = [
            f"datasets/miscellaneous/test/{domain}/" for n in selected_image_indices for domain in domain_folders
        ]

        image_paths = [[folder + f"{n}.png" for folder in domain_folders] for n in selected_image_indices]

        # loads the desired images
        examples = [tuple(load_image(path) for path in example) for example in image_paths]
        examples = [example for example in examples if not None in example]

        # generates the images
        for n, images in enumerate(tqdm(examples, desc=f"Generating images from batch {b}", position=1, leave=False)):
            star_model.generate_all_to_all(images, save_name=f'all-to-all/{selected_image_indices[n]}.png')
