# Accelerated Magnetic Resonance Imaging with Flow-Based Priors
This repository contains code for the master's thesis [Accelerated Magnetic Resonance Imaging with Flow-Based Priors](https://www.dropbox.com/s/t25jdo0gvc1z7gl/Accelerated%20Magnetic%20Resonance%20Imaging%20with%20Flow-Based%20Priors.pdf?dl=0).
In the thesis, we first train a flow-based generator on patches of MR images and then impose it as a prior for reconstructing an image from undersampled MRI measurements.

Below, we outline the usage of three components:
* Reconstructing a single image.
* Evaluating on an entire dataset.
* Training a prior from scratch.

## Installation
1. Install [PyTorch](https://pytorch.org/get-started/locally/) (preferably with an appropriate CUDA setup).
2. Install our package and all its requirements by running `pip install -e .` from the main `flow_based_priors` directory.
3. Install the [BART toolbox](https://mrirecon.github.io/bart/).
4. Download the [fastMRI dataset](https://fastmri.org/dataset/).
5. Point `get_bart_path()` and `get_mri_data_path()` in `kernprior/paths.py` to the respective paths on your machine.

## Usage
The scripts automatically download trained checkpoints of [Glow](https://arxiv.org/abs/1807.03039) models trained on the entire fastMRI multi-coil knee training set.
The code can be run on the GPU by providing `local_rank` as a command line argument. If `local_rank` is not specified, the CPU is used by default.

### Reconstructing a single image
Go to the `kernprior` directory and run:
```bash
python reconstruct.py --file_name FILE_NAME [--local_rank LOCAL_RANK]
```
where `FILE_NAME` is the name of a file in the fastMRI multi-coil knee validation set (e.g. `file1000831.h5`) and `LOCAL_RANK` is the index of the GPU to use.
Additional parameters can be set in `config/reconstruction.yaml`.

### Evaluating on an entire dataset
Go to the `kernprior` directory and run:
```bash
python evaluate.py --dataset_name DATASET_NAME [--local_rank LOCAL_RANK]
```
where `DATASET_NAME` is one of `knee_val`, `brain_100`, `stanford`, and `fastmri_a`. `LOCAL_RANK` is the index of the GPU to use.
These are the available datasets:
* `knee_val`: All the 199 mid-slices from the fastMRI multi-coil knee validation set.
* `brain_100`: 100 randomly selected mid-slices from the fastMRI multi-coil brain validation set.
* `stanford`: The [Stanford dataset](https://github.com/MLI-lab/Robustness-CS/tree/1fc9005fcc2841a4ebbd26f6c54d1f73d0648243#datasets) consisting of 18 multi-coil knee images.
* `fastmri_a`: The [fastMRI-A dataset](https://github.com/MLI-lab/Robustness-CS/tree/1fc9005fcc2841a4ebbd26f6c54d1f73d0648243#datasets) consisting of 105 adversarially-filtered multi-coil knee images.

The Stanford dataset is automatically downloaded as needed. The remaining datasets are subsets of the fastMRI dataset. 
The specific file names that constitute `brain_100` and `fastmri_a` can be found in the `data` directory.
Additional parameters can be set in `config/reconstruction.yaml`.

### Training a prior from scratch
Go to the `kernprior` directory and run:
```bash
python train_glow.py --acquisition ACQUISITION [--local_rank LOCAL_RANK]
```
where `ACQUISITION` is one of `including_fat` and `excluding_fat`. `including_fat` and `excluding_fat` correspond to the two acquisition types of knees in the fastMRI dataset. Additional parameters can be set in `config/training.yaml`.

## Testing
Go to the `test` directory and run:
```bash
python -m unittest
```

## Acknowledgments
The thesis is supervised by [Prof. Dr. Reinhard Heckel](http://reinhardheckel.com/).
The Glow implementation in `glow.py` and `train_glow.py` is taken from [here](https://github.com/kamenbliznashki/normalizing_flows).

## License
[MIT](https://github.com/ffraaz/flow_based_priors/blob/main/LICENSE)
