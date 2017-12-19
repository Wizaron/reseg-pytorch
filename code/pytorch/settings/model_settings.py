import os
import numpy as np
from data_settings import DataSettings

class ModelSettings(DataSettings):

    def __init__(self):
        super(ModelSettings, self).__init__()

        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.IMAGE_SIZE_HEIGHT = 800
        self.IMAGE_SIZE_WIDTH = 512
        self.ANNOTATION_SIZE_HEIGHT = 800
        self.ANNOTATION_SIZE_WIDTH = 512

        self.CROP_SCALE = (0.5, 1.0)
        self.CROP_AR = (3. / 4., 4. / 3.)
