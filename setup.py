from setuptools import setup

setup(name='kernprior',
      version='0.1.0',
      python_requires='>=3.7',
      packages=['kernprior'],
      install_requires=[
          'numpy>=1.19.2',
          'scikit_image>=0.18.1',
          'torchvision>=0.9.0',
          'torch>=1.8.0,<1.9',
          'h5py>=2.10.0',
          'PyYAML>=5.3.1',
          'scipy>=1.6.2',
          'fastmri==0.1.1',
          'packaging>=21.0'
      ])
