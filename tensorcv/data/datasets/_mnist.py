import os
import gzip
import time
import imageio
import numpy as np

def mnist_fashion(root):
    start = time.time()
    assert tf.gfile.IsDirectory(root), '`root` should be directory.'
    task_path = os.path.join(root, 'mnist_fashion')
    url_list = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz']
    if tf.gfile.Exists(task_path):
        tf.gfile.DeleteRecursively(task_path)
    tf.gfile.MakeDirs(task_path)
    for url in url_list:
        tf.keras.utils.get_file(os.path.join(task_path, url.split('/')[-1]), url)
    with gzip.open(os.path.join(task_path, 'train-labels-idx1-ubyte.gz'), 'rb') as lbpath:
        train_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(task_path, 'train-images-idx3-ubyte.gz'), 'rb') as imgpath:
        train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_label), 28, 28)

    with gzip.open(os.path.join(task_path, 't10k-labels-idx1-ubyte.gz'), 'rb') as lbpath:
        test_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(task_path, 't10k-images-idx3-ubyte.gz'), 'rb') as imgpath:
        test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(test_label), 28, 28)
    
    for i in set(train_label):
        tf.gfile.MakeDirs(os.path.join(task_path, 'train', str(i)))
    for i in set(test_label):
        tf.gfile.MakeDirs(os.path.join(task_path, 'test', str(i)))
    for idx in range(train.shape[0]):
        imageio.imsave(os.path.join(task_path, 'train', str(train_label[idx]), str(idx)+'.png'), train[idx])
    for idx in range(test.shape[0]):
        imageio.imsave(os.path.join(task_path, 'test', str(test_label[idx]), str(idx)+'.png'), test[idx])
    for url in url_list:
        tf.gfile.Remove(os.path.join(task_path, url.split('/')[-1]))
    print('mnist_fashion dataset download completed, run time %d min %.2f sec' %divmod((time.time()-start), 60))
    return task_path

