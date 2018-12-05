from io import open
from setuptools import setup, find_packages

def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

setup(name='kerascv',
      version='0.0.5',
      install_requires=['tensorflow>=1.12.0'],
      description='tf.Keras implementations of a Deep Learning Toolkit for Computer Vision.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/Hourout/keras-cv',
      author='JinQing Lee',
      author_email='hourout@163.com',
      keywords=['computer-vision', 'keras', 'tf.keras', 'deep-learning'],
      license='Apache-2.0',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: Apache-2.0',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
      packages=find_packages(),
      zip_safe=False)
