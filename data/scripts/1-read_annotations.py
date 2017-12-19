import os, glob
from PIL import Image
from utils import read_mat

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed', 'annotations')

try:
    os.makedirs(OUTPUT_DIR)
except:
    pass

annotation_files = glob.glob(os.path.join(DATA_DIR, 'raw', 'clothing-co-parsing', 'annotations', 'pixel-level', '*.mat'))

for f in annotation_files:
    image_name = os.path.splitext(os.path.basename(f))[0]

    annotation = read_mat(f)['groundtruth']
    annotation = Image.fromarray(annotation)
    annotation.save(os.path.join(OUTPUT_DIR, image_name + '.png'))
