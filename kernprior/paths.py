import os
from kernprior import utils


def get_bart_path() -> str:
    """Return the path to the main directory of the Bart toolbox."""
    return '/absolute/path/to/toolboxes/bart'


def get_mri_data_path(dataset: str, challenge: str, anatomy: str = 'knee') -> str:
    """Return the path to the fastMRI dataset. This path should point to a directory that contains HDF5 files."""
    return f'/absolute/path/to/data/{anatomy}/{challenge}_{dataset}/'


def get_data_path() -> str:
    return to_abs_path('data')


def get_results_path() -> str:
    return to_abs_path('results')


def get_model_path() -> str:
    return to_abs_path('models')


def get_config_path() -> str:
    return to_abs_path('config')


def get_data_file_path(dataset: str, challenge: str, file_name: str, anatomy: str = 'knee') -> str:
    return os.path.join(get_mri_data_path(dataset, challenge, anatomy), file_name)


def get_stanford_data_path() -> str:
    return os.path.join(get_data_path(), 'stanford')


def get_fastmri_a_path() -> str:
    return os.path.join(get_data_path(), 'fastmri_a.csv')


def get_brain_100_path() -> str:
    return os.path.join(get_data_path(), 'brain_100.txt')


def get_save_dir(file_name: str) -> str:
    return os.path.join(get_results_path(), file_name, utils.get_timestamp())


def get_evaluation_path(evaluation_name: str, file_name: str = '', slice_num: int = -1) -> str:
    path = os.path.join(get_results_path(), evaluation_name[:-20], evaluation_name[-19:])
    if file_name:
        path = os.path.join(path, file_name)

    if slice_num >= 0:
        path = os.path.join(path, f'slice_{slice_num}')

    return path


def get_sens_maps_dir(sens_params: str) -> str:
    base_dir = to_abs_path('sens_maps')
    sens_params = sens_params.replace('-', '_').replace(' ', '')
    sens_params = '_default' if sens_params == '' else sens_params
    sens_params = 'params' + sens_params
    return os.path.join(base_dir, sens_params)


def get_sens_file_path(file_name: str, slice_index: int, sens_params: str) -> str:
    return os.path.join(get_sens_maps_dir(sens_params), file_name, f'slice_{slice_index}.pt')


def to_abs_path(relative_path: str) -> str:
    return os.path.abspath(os.path.join('..', relative_path))
