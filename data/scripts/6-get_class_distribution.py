import os
import numpy as np
from PIL import Image

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
ANN_DIR = os.path.join(DATA_DIR, 'processed', 'annotations')
OUT_FILEPATH = os.path.join(DATA_DIR, 'metadata', 'class_weights.txt')

labels_filepath = os.path.join(DATA_DIR, 'metadata', 'labels.txt')
labels = np.loadtxt(labels_filepath, dtype='str', delimiter=',')
n_classes = len(labels)

lst_filepath = os.path.join(DATA_DIR, 'metadata', 'training.lst')
lst = np.loadtxt(lst_filepath, dtype='str', delimiter=' ')

class_dist = dict()
for image_name in lst:
    ann_path = os.path.join(ANN_DIR, image_name + '.png')
    img = Image.open(ann_path)
    img_np = np.array(img)
    img.close()
    classes = img_np.flatten()
    for c in classes:
        if not class_dist.has_key(c):
            class_dist[c] = 0
        class_dist[c] += 1

class_dist = np.array([[k, v] for k, v in class_dist.iteritems()], dtype='float')

class_dist[:, 1] = 1.0 / class_dist[:, 1]
n_total = class_dist[:, 1].sum()
class_dist[:, 1] = len(class_dist) * class_dist[:, 1] / n_total
class_dist = {k : v for k, v in class_dist}

class_dist_all = []
for i in range(n_classes):
    if class_dist.has_key(i):
        class_dist_all.append([i, class_dist[i]])
    else:
        class_dist_all.append([i, 0.0])

np.savetxt(OUT_FILEPATH, class_dist_all, fmt='%s', delimiter=',')
