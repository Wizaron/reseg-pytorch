import os
import numpy as np

RATIO = 0.9
np.random.seed(73)

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
image_names = np.loadtxt(os.path.join(DATA_DIR, 'metadata', 'image_shapes.txt'), dtype='str', delimiter=',')[:, 0]

np.random.shuffle(image_names)

n_training = int(round(RATIO * len(image_names)))
training_image_names = image_names[:n_training]
test_image_names = image_names[n_training:]

np.savetxt(os.path.join(DATA_DIR, 'metadata', 'training.lst'), training_image_names, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(DATA_DIR, 'metadata', 'test.lst'), test_image_names, fmt='%s', delimiter=' ')
