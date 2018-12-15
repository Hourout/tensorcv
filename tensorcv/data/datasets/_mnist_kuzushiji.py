import os
import time
import imageio
import numpy as np

def mnist_kuzushiji10(root):
    start = time.time()
    assert tf.gfile.IsDirectory(root), '`root` should be directory.'
    task_path = os.path.join(root, 'mnist_kuzushiji10')
    url_list = ['http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
                'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
                'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
                'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz']
    if tf.gfile.Exists(task_path):
        tf.gfile.DeleteRecursively(task_path)
    tf.gfile.MakeDirs(task_path)
    for url in url_list:
        tf.keras.utils.get_file(os.path.join(task_path, url.split('/')[-1]), url)
    train = np.load(os.path.join(task_path, 'kmnist-train-imgs.npz'))['arr_0']
    train_label = np.load(os.path.join(task_path, 'kmnist-train-labels.npz'))['arr_0']
    test = np.load(os.path.join(task_path, 'kmnist-test-imgs.npz'))['arr_0']
    test_label = np.load(os.path.join(task_path, 'kmnist-test-labels.npz'))['arr_0']
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
    print('mnist_kuzushiji10 dataset download completed, run time %d min %.2f sec' %divmod((time.time()-start), 60))
    return task_path

def mnist_kuzushiji49(root):
    start = time.time()
    assert tf.gfile.IsDirectory(root), '`root` should be directory.'
    task_path = os.path.join(root, 'mnist_kuzushiji49')
    url_list = ['http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
                'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
                'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
                'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz']
    if tf.gfile.Exists(task_path):
        tf.gfile.DeleteRecursively(task_path)
    tf.gfile.MakeDirs(task_path)
    for url in url_list:
        tf.keras.utils.get_file(os.path.join(task_path, url.split('/')[-1]), url)
    train = np.load(os.path.join(task_path, 'k49-train-imgs.npz'))['arr_0']
    train_label = np.load(os.path.join(task_path, 'k49-train-labels.npz'))['arr_0']
    test = np.load(os.path.join(task_path, 'k49-test-imgs.npz'))['arr_0']
    test_label = np.load(os.path.join(task_path, 'k49-test-labels.npz'))['arr_0']
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
    print('mnist_kuzushiji49 dataset download completed, run time %d min %.2f sec' %divmod((time.time()-start), 60))
    return task_path
