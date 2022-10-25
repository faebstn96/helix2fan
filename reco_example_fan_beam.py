import torch
from torch_radon import RadonFanbeam
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from helper import load_tiff_stack_with_metadata, save_to_tiff_stack


def run_reco(parser):
    projections, metadata = load_tiff_stack_with_metadata(Path(parser.path_proj))

    reco = []
    for i in range(projections.shape[2]):
        # Flip the projections to get reasonable reconstructions ("Siemens flip").
        prj = np.copy(np.flip(projections[:, :, i], axis=1))

        angles = np.array(metadata['angles'])[:metadata['rotview']] + (np.pi / 2)
        vox_scaling = 1 / parser.voxel_size

        radon = RadonFanbeam(parser.image_size,
                             angles,
                             source_distance=vox_scaling * metadata['dso'],
                             det_distance=vox_scaling * metadata['ddo'],
                             det_count=prj.shape[1],
                             det_spacing=vox_scaling * metadata['du'],
                             clip_to_circle=False)
        sinogram = torch.tensor(prj * vox_scaling).cuda()

        with torch.no_grad():
            filtered_sinogram = radon.filter_sinogram(sinogram, filter_name=parser.fbp_filter)
            fbp = radon.backprojection(filtered_sinogram)
            fbp = fbp.cpu().detach().numpy()
        reco.append(fbp)

    reco = np.array(reco)

    # Scale reconstruction to HU values following the DICOM-CT-PD
    # User Manual Version 3: WaterAttenuationCoefficient description
    fbp_hu = 1000 * ((reco - metadata['hu_factor']) / metadata['hu_factor'])

    save_path = Path(parser.path_out) / Path('{}_reco_{}.tif'.format(parser.scan_id, parser.fbp_filter))
    save_to_tiff_stack(fbp_hu, save_path)

    print('Reconstruction saved to {}.'.format(save_path))

    return fbp_hu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_proj', type=str, required=True, help='Local path of fan beam projection data.')
    parser.add_argument('--path_out', type=str, default='out', help='Output path of rebinned data.')
    parser.add_argument('--scan_id', type=str, default='Scan_001', help='Custom scan ID.')
    parser.add_argument('--image_size', type=int, default=512, help='Size of reconstructed image.')
    parser.add_argument('--voxel_size', type=float, default=0.7, help='In-slice voxel size [mm].')
    parser.add_argument('--fbp_filter', type=str, default='hann', nargs='?',
                        choices=['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'], help='Filter used for FBP.')

    fbp_hu = run_reco(parser)

    plt.imshow(fbp_hu[int(fbp_hu.shape[0] * 0.5)], cmap='gray', vmin=-500, vmax=300)
    plt.axis('off')
    plt.show()