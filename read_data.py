import numpy as np
import tqdm
import os
import pydicom
import struct


def unpack_tag(data, tag):
    return struct.unpack('f', data[tag].value)[0]


def read_projections(folder, indices):
    """ Read DICOM-CT-PD projection data following the DICOM-CT-PD User Manual Version 3
    of the TCIA LDCT-and-Projection-data.

    :param folder: full path to projection data folder
    :param indices: slice object that describes the range of helical projections to be loaded
    :return: An array containing the loaded DICOM projections to access header information, the raw projection data
    """
    datasets = []

    # Get the relevant file names.
    file_names = sorted([f for f in os.listdir(folder) if f.endswith(".dcm")])

    if len(file_names) == 0:
        raise ValueError('No DICOM files found in {}'.format(folder))

    file_names = file_names[indices]

    raw_projections = None

    for i, file_name in enumerate(tqdm.tqdm(file_names, 'Loading projection data')):
        # Read the DICOM projection.
        dataset = pydicom.dcmread(folder + '/' + file_name)

        # Get required information.
        rows = dataset.Rows
        cols = dataset.Columns
        # hu_factor = float(dataset[0x70411001].value)  # WaterAttenuationCoefficient, see manual for HU conversion.
        rescale_intercept = dataset.RescaleIntercept
        rescale_slope = dataset.RescaleSlope

        # Load the array as bytes.
        proj_array = np.array(np.frombuffer(dataset.PixelData, 'H'), dtype='float32')
        proj_array = proj_array.reshape([cols, rows], order='F')

        # Rescale array according to the TCIA (LDCT-and-Projection-data) DICOM-CT-PD User Manual Version 3.
        proj_array *= rescale_slope
        proj_array += rescale_intercept
        # proj_array /= hu_factor

        # Store results.
        if raw_projections is None:
            # We need to load the first dataset before we know the shape.
            raw_projections = np.empty((len(file_names), cols, rows), dtype='float32')

        raw_projections[i] = proj_array[:, ::-1]
        datasets.append(dataset)

    return datasets, raw_projections


def read_dicom(parser):
    """ Read DICOM-CT-PD projection data and header information following the DICOM-CT-PD User Manual Version 3
    of the TCIA LDCT-and-Projection-data.

    :param parser: parser containing projection information
    :return: the raw projection data, a parser that contains all relevant DICOM header
    parameters for rebinning and reconstruction
    """
    args = parser.parse_args()
    indices = slice(args.idx_proj_start, args.idx_proj_stop)
    data_headers, raw_projections = read_projections(args.path_dicom, indices)

    # Read geometry information from the DICOM headers following instructions from the
    # TCIA (LDCT-and-Projection-data) DICOM-CT-PD User Manual Version 3.
    angles = np.array([unpack_tag(d, 0x70311001) for d in data_headers]) + (np.pi / 2)
    angles = - np.unwrap(angles) - np.pi  # Different definition of angles (monotonously increasing, starting from a negative value)
    dangles = np.array([unpack_tag(d, 0x7033100B) for d in data_headers])  # Flying focal spot dphi
    dz = np.array([unpack_tag(d, 0x7033100C) for d in data_headers])  # Flying focal spot dz
    drho = np.array([unpack_tag(d, 0x7033100D) for d in data_headers])  # Flying focal spot drho
    nu = data_headers[0].Rows
    nv = data_headers[0].Columns
    du = unpack_tag(data_headers[0], 0x70291002)  # DetectorElementTransverseSpacing
    dv = unpack_tag(data_headers[0], 0x70291006)  # DetectorElementAxialSpacing
    dv_rebinned = 1  # [mm] Detector pixel v width of rebinned sinogram.
    det_central_element = np.array(struct.unpack('2f', data_headers[0][0x70311033].value))
    dso = unpack_tag(data_headers[0], 0x70311003)  # DetectorFocalCenterRadialDistance
    dsd = unpack_tag(data_headers[0], 0x70311031)  # ConstantRadialDistance
    ddo = (unpack_tag(data_headers[0], 0x70311031) - unpack_tag(data_headers[0], 0x70311003))  # ConstantRadialDistance - DetectorFocalCenterRadialDistance
    pitch = ((unpack_tag(data_headers[-1], 0x70311002) -
              unpack_tag(data_headers[0], 0x70311002)) /
             ((np.max(angles) - np.min(angles)) / (2 * np.pi)))  # Mayo does not include the tag TableFeedPerRotation, we manually compute the pitch
    z_positions = np.array([unpack_tag(d, 0x70311002) for d in data_headers])  # DetectorFocalCenterAxialPosition
    nz_rebinned = int((z_positions[-1] - z_positions[0]) / dv_rebinned)
    hu_factor = float(data_headers[0][0x70411001].value)  # WaterAttenuationCoefficient (see manual for HU conversion)
    rotview = int(len(angles) / ((angles[-1] - angles[0]) / (2 * np.pi)))

    # Create parser.
    parser.add_argument('--indices', type=int, default=[indices.start, indices.stop],
                        help='Index range of loaded and processed helical projections.')
    parser.add_argument('--nu', type=int, default=nu,
                        help='Number of scanner detector rows [].')
    parser.add_argument('--nv', type=int, default=nv,
                        help='Number of scanner detector columns [].')
    parser.add_argument('--du', type=float, default=du,
                        help='Scanner detector pixel spacing in row direction [mm].')
    parser.add_argument('--dv', type=float, default=dv,
                        help='Scanner detector pixel spacing in column direction [mm].')
    parser.add_argument('--dv_rebinned', type=float, default=dv_rebinned,
                        help='Detector pixel spacing in column direction on virtual rebinned detector [mm].')
    parser.add_argument('--det_central_element', type=float, default=det_central_element.tolist(),
                        help='Central element of the detector. Float index between real detector pixels [].')
    parser.add_argument('--dso', type=float, default=dso,
                        help='Source-object (isocenter) distance [mm].')
    parser.add_argument('--dsd', type=float, default=dsd,
                        help='Source-detector distance [mm].')
    parser.add_argument('--ddo', type=float, default=ddo,
                        help='Detector-object distance [mm].')
    parser.add_argument('--pitch', type=float, default=pitch,
                        help='Pitch of helical trajectory [mm].')
    parser.add_argument('--nz_rebinned', type=int, default=nz_rebinned,
                        help='Number of detector rows on the rebinned detector.')
    parser.add_argument('--rotview', type=int, default=rotview,
                        help='Number of projections of rebinned detector, i.e., number of helical projections within [0, 2pi].')
    parser.add_argument('--hu_factor', type=float, default=hu_factor,
                        help='Water attenuation coefficient to convert the absorption coefficients to HU values [].')
    parser.add_argument('--angles', type=float, default=angles.tolist(),
                        help='Angles of helix projections [rad].')
    parser.add_argument('--z_positions', type=float, default=z_positions.tolist(),
                        help='Axial positions of projections of the helical trajectory [mm].')
    parser.add_argument('--dangles', type=float, default=dangles.tolist(),
                        help='Flying focal spot correction dphi [rad].')
    parser.add_argument('--dz', type=float, default=dz.tolist(),
                        help='Flying focal spot correction dz [mm].')
    parser.add_argument('--drho', type=float, default=drho.tolist(),
                        help='Flying focal spot correction drho [mm].')

    return raw_projections, parser
