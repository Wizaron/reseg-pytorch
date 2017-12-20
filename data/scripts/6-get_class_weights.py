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

classes_in_annotations = []; annotation_sizes = []
class_dist = dict()
for image_name in lst:
    ann_path = os.path.join(ANN_DIR, image_name + '.png')
    img = Image.open(ann_path)
    img_np = np.array(img)
    img.close()

    classes_in_annotations.append(np.unique(img_np))
    annotation_sizes.append(img_np.size)

    classes = img_np.flatten()
    for c in classes:
        if not class_dist.has_key(c):
            class_dist[c] = 0
        class_dist[c] += 1

class_dist = np.array([[k, v] for k, v in class_dist.iteritems()], dtype='int')
classes = class_dist[:, 0]
class_counts = class_dist[:, 1]

class_size_counts = np.zeros(classes.shape)
for c, s in zip(classes_in_annotations, annotation_sizes):
    for cc in c:
        class_size_counts[cc] += s

priors = class_counts.astype('float') / class_size_counts
w_freq = np.median(priors) / priors

class_dist = {c : w for c, w in zip(classes, w_freq)}

class_dist_all = []
for i in range(n_classes):
    if class_dist.has_key(i):
        class_dist_all.append([i, class_dist[i]])
    else:
        class_dist_all.append([i, 0.0])

np.savetxt(OUT_FILEPATH, class_dist_all, fmt='%s', delimiter=',')
