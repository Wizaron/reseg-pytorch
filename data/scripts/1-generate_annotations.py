import os, glob
from PIL import Image
import numpy as np
from scipy.io import loadmat

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'VOCdevkit', 'VOC2010', 'JPEGImages')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed', 'annotations')

try:
    os.makedirs(OUTPUT_DIR)
except:
    pass

ann_files = glob.glob(os.path.join(DATA_DIR, 'raw', 'Annotations_Part', '*.mat'))

for ann_file in ann_files:
    img_name = os.path.splitext(os.path.basename(ann_file))[0]

    anns = loadmat(ann_file)['anno'][0][0][1][0]

    persons = [[ann[2], ann[3]] for ann in anns if ann[0][0] == 'person']

    if len(persons) == 0:
        continue

    img = Image.open(os.path.join(IMG_DIR, img_name + '.jpg'))
    img_width, img_height = img.size

    head_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for person in persons:
        #person[0] -> person mask
        #person[1] -> parts

        if len(person[1]) == 0:
            continue

        parts = person[1][0]

        for part in parts:
            part_name, part_ann = part
            part_name = part_name[0]
            if part_name in ['head', 'hair']:
                head_mask[part_ann == 1] = 1

    head_mask_img = Image.fromarray(head_mask)

    head_mask_img.save(os.path.join(OUTPUT_DIR, img_name + '.png'))
