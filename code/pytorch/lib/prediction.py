import torch.nn as nn
import numpy as np
from PIL import Image

from utils import ImageUtilities

class Prediction(object):

    def __init__(self, resize_height, resize_width, mean, std, model):
        self.normalizer = ImageUtilities.image_normalizer(mean, std)

        self.resize_height = resize_height
        self.resize_width = resize_width
        self.model = model

    def get_image(self, image_path):
        img = ImageUtilities.read_image(image_path)

        image_width, image_height = img.size

        #assert image_height >= image_width
        #image_ar = float(image_height) / image_width
        #new_height = int(round(self.resize_width * image_ar))
        #new_width = int(round(self.resize_width))

        new_height = self.resize_height
        new_width = self.resize_width

        resizer = ImageUtilities.image_resizer(new_height, new_width)

        img = resizer(img)
        img = self.normalizer(img)
        return img, image_height, image_width

    def get_annotation(self, annotation_path):
        img = ImageUtilities.read_image(annotation_path)
        return img

    def upsample_prediction(self, prediction, image_height, image_width):

        #assert len(prediction.size()) == 4   # n, c, h, w  

        #return nn.UpsamplingNearest2d((image_height, image_width))(prediction)
        resizer = ImageUtilities.image_resizer(image_height, image_width, interpolation=Image.NEAREST)
        return resizer(prediction)

    def predict(self, image_path):

        image, image_height, image_width = self.get_image(image_path)
        image = image.unsqueeze(0)

        prediction = self.model.predict(image)
        #prediction = self.upsample_prediction(prediction, image_height, image_width)
        prediction = prediction.squeeze(0)
        prediction = prediction.data.cpu().numpy()
        prediction = prediction.argmax(0).astype(np.uint8) #TODO: max 255 classes
        prediction_pil = Image.fromarray(prediction)
        prediction_pil = self.upsample_prediction(prediction_pil, image_height, image_width)

        image_pil = ImageUtilities.read_image(image_path)
        #prediction_np = prediction.data.cpu().numpy()
        

        return image_pil, prediction_pil
