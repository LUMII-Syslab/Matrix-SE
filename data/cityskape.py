import glob
from pathlib import Path
from zipfile import ZipFile

import tensorflow as tf

import config
from data.base import Dataset

#  Fine annotation: https://www.cityscapes-dataset.com/file-handling/?packageID=1
#  Coarse annotation: https://www.cityscapes-dataset.com/file-handling/?packageID=2
#  Train-Val-Test set: https://www.cityscapes-dataset.com/file-handling/?packageID=3
#  Train extra set: https://www.cityscapes-dataset.com/file-handling/?packageID=4


FINE_ANNOTATIONS = "gtFine_trainvaltest.zip"
COARSE_ANNOTATIONS = "gtCoarse.zip"
BASE_DATASET = "leftImg8bit_trainvaltest.zip"
EXTRA_DATASET = "leftImg8bit_trainextra.zip"

AUTOTUNE = tf.data.experimental.AUTOTUNE


def extract_name(path):
    path = path.split("/")
    file = path[-1].split('_')[:3]
    mode = path[-3]
    return mode + ''.join(file)


def get_files(folder, files):
    train = glob.glob(folder + 'train' + files)
    val = glob.glob(folder + 'val' + files)
    # test = glob.glob(folder + 'test' + files)
    return train, val


def extract_examples(features: iter, labels: iter):
    name_to_path = {extract_name(path): path for path in labels}
    return [(path, name_to_path[extract_name(path)]) for path in features]


def read_images(feature, label):
    feature = tf.io.read_file(feature)
    feature = tf.image.decode_png(feature, channels=3)
    feature = tf.image.convert_image_dtype(feature, tf.float32)

    label = tf.io.read_file(label)
    label = tf.image.decode_png(label, channels=1)
    label = tf.cast(label, tf.int32)
    label_mapping = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0,
                                 0, 0, 6, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 16, 0, 0, 17, 18, 19, 0])
    label = tf.gather(label_mapping, label)

    return feature, label


def random_left_flip(feature, label):
    do_flip = tf.random.uniform(()) > 0.5
    feature = tf.cond(do_flip, lambda: tf.image.flip_left_right(feature), lambda: feature)
    label = tf.cond(do_flip, lambda: tf.image.flip_left_right(label), lambda: label)
    return feature, label


def resize_image(feature, label, size):
    feature = tf.image.resize(feature, [size, size * 2], tf.image.ResizeMethod.AREA, align_corners=True)
    label = tf.image.resize(label, [size, size * 2], tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    label = tf.squeeze(label, axis=2)

    return feature, label


def random_crop(feature, label):
    shape = tf.shape(feature)
    height = shape[0]
    width = shape[1]

    w_size = tf.random.uniform((), minval=512, maxval=width, dtype=tf.int32)

    h_prop = w_size // 2
    delta = h_prop // 5
    h_min = h_prop - delta
    h_max = tf.math.minimum(h_prop + delta, height)

    h_size = tf.random.uniform((), minval=h_min, maxval=h_max, dtype=tf.int32)

    w_offset = tf.random.uniform((), maxval=width - w_size, dtype=tf.int32)
    h_offset = tf.random.uniform((), maxval=height - h_size, dtype=tf.int32)

    do_crop = tf.random.uniform(()) > 0.5

    feature = tf.cond(do_crop, lambda: tf.image.crop_to_bounding_box(feature, h_offset, w_offset, h_size, w_size),
                      lambda: feature)
    label = tf.cond(do_crop, lambda: tf.image.crop_to_bounding_box(label, h_offset, w_offset, h_size, w_size),
                    lambda: label)

    return feature, label


def random_brightness(feature, label):
    feature = tf.image.random_brightness(feature, 0.5)
    return feature, label


def random_contrast(feature, label):
    feature = tf.image.random_contrast(feature, 0.3, 0.9)
    return feature, label


class CityScape(Dataset):

    def __init__(self) -> None:
        super().__init__({
            "input_classes": None,
            "output_classes": 20,
            "train_lengths": [512],
            "eval_lengths": [512],
            "use_coarse_dataset": True
        })
        self.location = f"{config.data_dir}/cityscape"

        self.unzip_archive(BASE_DATASET)
        self.unzip_archive(FINE_ANNOTATIONS)

        train_labels, test_labels = self.get_labels_fine()
        train_features, test_features = self.get_features_fine()

        self.train_examples = extract_examples(train_features, train_labels)
        self.test_examples = extract_examples(test_features, test_labels)

        if self.config["use_coarse_dataset"]:
            self.unzip_archive(COARSE_ANNOTATIONS)
            self.unzip_archive(EXTRA_DATASET)

            c_train_labels, c_test_labels = self.get_labels_coarse()
            c_train_features, c_test_features = self.get_features_coarse()

            self.train_examples += extract_examples(c_train_features, c_train_labels)
            # self.test_examples += extract_examples(c_test_features, c_test_labels)

    def get_features_fine(self):
        folder = self.location + '/leftImg8bit_trainvaltest/leftImg8bit/'
        files = '/*/*.png'
        return get_files(folder, files)

    def get_labels_fine(self) -> tuple:
        folder = self.location + '/gtFine_trainvaltest/gtFine/'
        files = '/*/*labelIds.png'
        return get_files(folder, files)

    def get_features_coarse(self):
        folder = self.location + '/leftImg8bit_trainvaltest/leftImg8bit/'
        folder_extra = self.location + '/leftImg8bit_trainextra/leftImg8bit/'

        files = '/*/*.png'

        train_extra = glob.glob(folder_extra + 'train_extra' + files)
        train = glob.glob(folder + 'train' + files)
        val = glob.glob(folder + 'val' + files)

        # "troisdorf_000000_000073_leftImg8bit.png" is corrupt/black
        error = glob.glob(folder_extra + 'train_extra' + "/*/troisdorf_000000_000073_*.png")
        train_extra.remove(error[0])

        return train_extra + train, val

    def get_labels_coarse(self) -> tuple:
        folder = self.location + '/gtCoarse/gtCoarse/'
        files = '/*/*labelIds.png'

        train_extra = glob.glob(folder + 'train_extra' + files)
        train = glob.glob(folder + 'train' + files)
        val = glob.glob(folder + 'val' + files)

        error = glob.glob(folder + 'train_extra' + "/*/troisdorf_000000_000073_*_labelIds.png")
        train_extra.remove(error[0])

        return train_extra + train, val

    def unzip_archive(self, zip_file):
        zip_path = Path(self.location) / zip_file
        folder_path = Path(self.location) / zip_path.name.split('.')[0]

        if not folder_path.exists():
            with ZipFile(str(zip_path), 'r') as archive:
                archive.extractall(str(folder_path))

    def train_dataset(self) -> list:
        dataset = self.create_dataset(self.train_examples)
        dataset = self.augment_dataset(dataset)
        return self.resize_datasets(dataset, self.config["train_lengths"])

    def eval_dataset(self) -> list:
        dataset = self.create_dataset(self.test_examples)
        return self.resize_datasets(dataset, self.config["eval_lengths"])

    @staticmethod
    def create_dataset(examples: list):
        features, labels = zip(*examples)
        features = tf.convert_to_tensor(features, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.string)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(50000)
        dataset = dataset.map(read_images, num_parallel_calls=AUTOTUNE)

        return dataset

    @staticmethod
    def resize_datasets(dataset, sizes):
        return [dataset.map(lambda x, y: resize_image(x, y, size), num_parallel_calls=AUTOTUNE) for size in sizes]

    @staticmethod
    def augment_dataset(dataset):
        dataset = dataset.map(random_left_flip, num_parallel_calls=AUTOTUNE)
        return dataset.map(random_contrast, num_parallel_calls=AUTOTUNE)
