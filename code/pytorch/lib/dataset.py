import torch
from torch.utils.data import Dataset
import random

from PIL import Image
import lmdb, sys
import numpy as np
from StringIO import StringIO

from utils import ImageUtilities

class SegDataset(Dataset):
    """Dataset Reader"""

    def __init__(self, lmdb_path):

        self._lmdb_path = lmdb_path

        self.env = lmdb.open(self._lmdb_path, max_readers=1, readonly=True,
                             lock=False, readahead=False, meminit=False)

        if not self.env:
            print 'Cannot read lmdb from {}'.format(self._lmdb_path)
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.n_samples = int(txn.get('num-samples'))

    def __load_data(self, index):

        with self.env.begin(write=False) as txn:
            image_key = 'image-{}'.format(index + 1)
            annotation_key = 'annotation-{}'.format(index + 1)
            img = txn.get(image_key)
            img = Image.open(StringIO(img))

            annotation = txn.get(annotation_key)
            annotation = Image.open(StringIO(annotation))

        return img, annotation

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'

        image, annotation = self.__load_data(index)

        return image, annotation

    def __len__(self):
        return self.n_samples

class AlignCollate(object):
    """Should be a callable (https://docs.python.org/2/library/functions.html#callable), that gets a minibatch
    and returns minibatch."""

    def __init__(self, mode, labels, mean, std, image_size_height, image_size_width, annotation_size_height, annotation_size_width,
                 crop_scale, crop_ar, random_cropping=True, horizontal_flipping=True):

        self._mode = mode

        assert self._mode in ['training', 'test']

        self.n_classes = len(labels)
        self.mean = mean
        self.std = std
        self.image_size_height = image_size_height
        self.image_size_width = image_size_width
        self.random_cropping = random_cropping
        self.crop_scale = crop_scale
        self.crop_ar = crop_ar
        self.annotation_size_height = annotation_size_height
        self.annotation_size_width = annotation_size_width
        self.horizontal_flipping = horizontal_flipping

        if self._mode == 'training':
            if self.random_cropping:
                self.image_random_cropper = ImageUtilities.image_random_cropper_and_resizer(self.image_size_height, self.image_size_width)
                self.annotation_random_cropper = ImageUtilities.image_random_cropper_and_resizer(self.annotation_size_height,
                                                                                                 self.annotation_size_width,
                                                                                                 interpolation=Image.NEAREST)
            else:
                self.image_resizer = ImageUtilities.image_resizer(self.image_size_height, self.image_size_width)
                self.annotation_resizer = ImageUtilities.image_resizer(self.annotation_size_height, self.annotation_size_width,
                                                                       interpolation=Image.NEAREST)
            if self.horizontal_flipping:
                self.horizontal_flipper = ImageUtilities.image_random_horizontal_flipper()
        else:
            self.image_resizer = ImageUtilities.image_resizer(self.image_size_height, self.image_size_width)
            self.annotation_resizer = ImageUtilities.image_resizer(self.annotation_size_height, self.annotation_size_width,
                                                                   interpolation=Image.NEAREST)

        self.image_normalizer = ImageUtilities.image_normalizer(self.mean, self.std)

    def __preprocess(self, image, annotation):

        if self._mode == 'training':
            if self.random_cropping:
                crop_params = self.image_random_cropper.get_params(image, scale=self.crop_scale, ratio=self.crop_ar)
                image = self.image_random_cropper(image, crop_params)
                annotation = self.annotation_random_cropper(annotation, crop_params)
            else:
                image = self.image_resizer(image)
                annotation = self.annotation_resizer(annotation)

            if self.horizontal_flipping:
                is_flip = random.random() < 0.5
                image = self.horizontal_flipper(image, is_flip)
                annotation = self.horizontal_flipper(annotation, is_flip)
        else:
            image = self.image_resizer(image)
            annotation = self.annotation_resizer(annotation)            

        image = self.image_normalizer(image)
        annotation = np.array(annotation)

        return image, annotation

    def __call__(self, batch):
        images, annotations = zip(*batch)
        images = list(images)
        annotations = list(annotations)

        bs = len(images)
        for i in range(bs):
            image, annotation = self.__preprocess(images[i], annotations[i])
            images[i] = image
            annotations[i] = annotation

        images = torch.stack(images)

        annotations = np.array(annotations, dtype='int')
        annotations_one_hot = np.eye(self.n_classes, dtype='int')
        annotations_one_hot = annotations_one_hot[annotations.flatten()].reshape(annotations.shape[0], annotations.shape[1],
                                                                                 annotations.shape[2], self.n_classes)
        annotations_one_hot = torch.LongTensor(annotations_one_hot)

        return images, annotations_one_hot
