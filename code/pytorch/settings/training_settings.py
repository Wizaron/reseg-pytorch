import os
from model_settings import ModelSettings

class TrainingSettings(ModelSettings):

    def __init__(self):
        super(TrainingSettings, self).__init__()

        self.TRAINING_LMDB = os.path.join(self.BASE_PATH, 'data', 'processed', 'lmdb', 'training-lmdb')
        self.VALIDATION_LMDB = os.path.join(self.BASE_PATH, 'data', 'processed', 'lmdb', 'test-lmdb')

        self.TRAIN_CNN = True

        self.OPTIMIZER = 'Adadelta'   # one of : 'RMSprop', 'Adam', 'Adadelta', 'SGD'
        self.LEARNING_RATE = 1.0
        self.LR_DROP_FACTOR = 0.1
        self.LR_DROP_PATIENCE = 10
        self.WEIGHT_DECAY = 0.001      # use 0 to disable it
        self.CLIP_GRAD_NORM = 0.0      # max l2 norm of gradient of parameters - use 0 to disable it

        self.CRITERION = 'Multi'      # One of 'CE', 'Dice', 'Multi'

        self.SEED = 13
