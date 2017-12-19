import os
import numpy as np
from utils import read_mat

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
OUTPUT_DIR = os.path.join(DATA_DIR, 'metadata')

labels_filepath = os.path.join(DATA_DIR, 'raw', 'clothing-co-parsing', 'label_list.mat')

labels = read_mat(labels_filepath)['label_list'][0]
labels = np.array([l[0] for i, l in enumerate(labels)])
labels = np.stack((range(len(labels)), labels), axis=1)

np.savetxt(os.path.join(OUTPUT_DIR, 'labels.txt'), labels, fmt='%s', delimiter=',')
