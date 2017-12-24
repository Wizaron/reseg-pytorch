import os, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='path of the image')
parser.add_argument('--model', required=True, help='path of the model')
parser.add_argument('--usegpu', action='store_true', help='enables cuda to predict on gpu')
parser.add_argument('--output', required=True, help='path of the output directory')
opt = parser.parse_args()

image_path = opt.image
model_path = opt.model
output_path = opt.output

try:
    os.makedirs(output_path)
except:
    pass

model_dir = os.path.dirname(model_path)
sys.path.insert(0, model_dir)

from lib import Model, Prediction
from settings import ModelSettings

ms = ModelSettings()

model = Model(ms.LABELS, load_model_path=model_path, usegpu=opt.usegpu)
prediction = Prediction(ms.IMAGE_SIZE_HEIGHT, ms.IMAGE_SIZE_WIDTH, ms.MEAN, ms.STD, model)
image, pred = prediction.predict(image_path)

image_name = os.path.splitext(os.path.basename(image_path))[0]
image.save(os.path.join(output_path, image_name + '.png'))
pred.save(os.path.join(output_path, image_name + '-mask.png'))
