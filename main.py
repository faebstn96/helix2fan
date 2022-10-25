import numpy as np
import argparse
import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from joblib import Memory
from helper import save_to_tiff_stack_with_metadata, load_tiff_stack_with_metadata
from rebinning_functions import rebin_curved_to_flat_detector, rebin_helical_to_fan_beam_trajectory, \
    _rebin_curved_to_flat_detector_core, rebin_curved_to_flat_detector_multiprocessing
from read_data import read_dicom


def run(parser):
    print('Processing scan {}.'.format(parser.scan_id))

    # Load projections and read out geometry data from the DICOM header.
    raw_projections, parser = read_dicom(parser)
    args = parser.parse_args()

    if parser.save_all:
        save_path = Path(parser.path_out) / Path('{}_curved_helix_projections.tif'.format(parser.scan_id))
        save_to_tiff_stack_with_metadata(raw_projections,
                                         save_path,
                                         metadata=vars(args))

    # Rebin helical projections from curved detector to flat detector.
    # Step can be skipped if the reconstruction supports curved detectors.
    if parser.no_multiprocessing:
        proj_flat_detector = rebin_curved_to_flat_detector(args, raw_projections)
    else:
        location = 'tmp/cache_dir'  # Todo: '/home/fabian/Desktop/tmp/cache_dir'
        memory = Memory(location, verbose=0)

        cached_rebin_curved_to_flat_detector_core = memory.cache(_rebin_curved_to_flat_detector_core)
        data = (args, raw_projections)
        proj_flat_detector = np.array(Parallel(n_jobs=8)(
            delayed(rebin_curved_to_flat_detector_multiprocessing)(data, col)
            for col in tqdm.tqdm(range(data[1].shape[0]), 'Rebin curved to flat detector')))

    if parser.save_all:
        save_path = Path(parser.path_out) / Path('{}_flat_helix_projections.tif'.format(parser.scan_id))
        save_to_tiff_stack_with_metadata(proj_flat_detector,
                                         save_path,
                                         metadata=vars(args))

    # Rebinning of projections acquired on a helical trajectory to full-scan (2pi) fan beam projections.
    proj_fan_geometry = rebin_helical_to_fan_beam_trajectory(args, proj_flat_detector)

    save_path = Path(parser.path_out) / Path('{}_flat_fan_projections.tif'.format(parser.scan_id))
    save_to_tiff_stack_with_metadata(proj_fan_geometry,
                                     save_path,
                                     metadata=vars(args))

    print('Finished. Results saved at {}.'.format(save_path.resolve()))


if __name__ == '__main__':

    path_dicom = '/media/fabian/HD0/Data/DATABASE_Cancer_Archive/projection_data_2022-09-27/L277/08-27-2018-NA-NA-06585/1.000000-Low dose projections-15901'

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dicom', type=str, required=True, help='Local path of helical projection data.')
    parser.add_argument('--path_out', type=str, default='out', help='Output path of rebinned data.')
    parser.add_argument('--scan_id', type=str, default='Scan_001', help='Custom scan ID.')
    parser.add_argument('--idx_proj_start', type=int, default=12000, help='First index of helical projections that are processed.')
    parser.add_argument('--idx_proj_stop', type=int, default=26000, help='Last index of helical projections that are processed.')
    parser.add_argument('--save_all', dest='save_all', action='store_true', help='Save intermediate results.')
    parser.add_argument('--no_multiprocessing', dest='save_all', action='store_true', help='Switch off multiprocessing using joblib.')
    run(parser)