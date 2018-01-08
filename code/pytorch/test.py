import argparse, os, sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='path of model')
parser.add_argument('--lmdb', required=True, help='path of lmdb')
parser.add_argument('--usegpu', action='store_true', help='enables cuda to train on gpu')
parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
parser.add_argument('--nworkers', type=int, help='number of data loading workers [0 to do it using main process]', default=0)
opt = parser.parse_args()

model_path = opt.model

model_dir = os.path.dirname(model_path)
sys.path.insert(0, model_dir)

from lib import SegDataset, Model, AlignCollate
from settings import ModelSettings

ms = ModelSettings()

if torch.cuda.is_available() and not opt.usegpu:
    print('WARNING: You have a CUDA device, so you should probably run with --cuda')

# Define Data Loaders
pin_memory = False
if opt.usegpu:
    pin_memory = True

test_dataset = SegDataset(opt.lmdb)
test_align_collate = AlignCollate('test', ms.LABELS, ms.MEAN, ms.STD, ms.IMAGE_SIZE_HEIGHT, ms.IMAGE_SIZE_WIDTH,
                                  ms.ANNOTATION_SIZE_HEIGHT, ms.ANNOTATION_SIZE_WIDTH, ms.CROP_SCALE, ms.CROP_AR,
                                  random_cropping=ms.RANDOM_CROPPING, horizontal_flipping=ms.HORIZONTAL_FLIPPING)
assert test_dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False,
                                          num_workers=opt.nworkers, pin_memory=pin_memory, collate_fn=test_align_collate)

# Define Model
model = Model(ms.LABELS, load_model_path=model_path, usegpu=opt.usegpu)

# Test Model
test_accuracy, test_dice_coeff = model.test(ms.CLASS_WEIGHTS, test_loader)
