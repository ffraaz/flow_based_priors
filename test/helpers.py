from kernprior import utils
from kernprior import paths


def get_kspace(challenge):
    file_path = paths.get_data_file_path('val', challenge, 'file1000831.h5')
    return utils.load_kspace_slice(file_path)


def get_num_mid_slices(mid_slice_range):
    return 2 * mid_slice_range + 1
