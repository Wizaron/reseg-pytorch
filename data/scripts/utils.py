import scipy.io
import lmdb

def read_mat(filepath):
    return scipy.io.loadmat(filepath)

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)

def create_dataset(output_path, image_paths, annotation_paths):

    n_images = len(image_paths)

    assert(n_images == len(annotation_paths))

    print 'Number of Images : {}'.format(n_images)

    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    n_images_cntr = 1
    for i in xrange(n_images):
        image_path = image_paths[i]
        annotation_path = annotation_paths[i]

        image = open(image_path, 'r').read()
        annotation = open(annotation_path, 'r').read()

        cache['image-{}'.format(n_images_cntr)] = image
        cache['annotation-{}'.format(n_images_cntr)] = annotation

        if n_images_cntr % 500 == 0:
            write_cache(env, cache)
            cache = {}
            print 'Processed %d / %d' % (n_images_cntr, n_images)
        n_images_cntr += 1

    cache['num-samples'] = str(n_images)
    write_cache(env, cache)
    print 'Created dataset with {} samples'.format(n_images)
