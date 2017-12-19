from PIL import Image
import torchvision.transforms as transforms

from preprocess import RandomResizedCrop, RandomHorizontalFlip

class ImageUtilities(object):

    @staticmethod
    def read_image(image_path):
        img = Image.open(image_path)
        img_copy = img.copy()
        img.close()
        return img_copy

    @staticmethod
    def image_resizer(height, width, interpolation=Image.BILINEAR):
        return transforms.Resize((height, width), interpolation=interpolation)

    @staticmethod
    def image_random_cropper_and_resizer(height, width, interpolation=Image.BILINEAR):
        return RandomResizedCrop(height, width, interpolation=interpolation)

    @staticmethod
    def image_random_horizontal_flipper():
        return RandomHorizontalFlip()

    @staticmethod
    def image_normalizer(mean, std):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
