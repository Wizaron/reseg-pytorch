import os
import numpy as np
from data_settings import DataSettings

class ModelSettings(DataSettings):

    def __init__(self):
        super(ModelSettings, self).__init__()

        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.IMAGE_SIZE_HEIGHT = 827
        self.IMAGE_SIZE_WIDTH = 550
        self.ANNOTATION_SIZE_HEIGHT = 824
        self.ANNOTATION_SIZE_WIDTH = 548

        self.HORIZONTAL_FLIPPING = True
        self.RANDOM_CROPPING = True       # CROP_SCALE and CROP_AR is used iff self.RANDOM_CROPPING is True
        self.CROP_SCALE = (0.8, 1.0)      # Choose it carefully - have a look at lib/preprocess.py -> RandomResizedCrop
        self.CROP_AR = (3. / 4., 4. / 3.) # Choose it carefully - have a look at lib/preprocess.py -> RandomResizedCrop
