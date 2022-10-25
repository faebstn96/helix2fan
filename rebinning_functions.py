import numpy as np
import tqdm


def interpolate_2d(grid, x, y):
    """ 2D linear interpolation of location (x, y) on grid.

    :param grid: image grid where (x, y) is to be interpolated on
    :param x: x coordinate of interpolated location
    :param y: y coordinate of interpolated location
    :return: interpolated intensity value at (x, y)
    """
    # Calculate the four surrounding data points in range [0, 1].
    x_floor = int(x)
    y_floor = int(y)
    x_floor_plus_one = x_floor + 1
    y_floor_plus_one = y_floor + 1
    x_p = x - x_floor
    y_p = y - y_floor

    # Check boundary conditions.
    if x_floor < 0 or x_floor > grid.shape[0] - 1:
        x_floor = None
    if x_floor_plus_one < 0 or x_floor_plus_one > grid.shape[0] - 1:
        x_floor_plus_one = None

    if y_floor < 0 or y_floor > grid.shape[1] - 1:
        y_floor = None
    if y_floor_plus_one < 0 or y_floor_plus_one > grid.shape[1] - 1:
        y_floor_plus_one = None

    # Get function values of the data points and setup function matrix.
    a = grid[x_floor, y_floor] if (x_floor is not None and y_floor is not None) else 0.0
    b = grid[x_floor, y_floor_plus_one] if \
        (x_floor is not None and y_floor_plus_one is not None) else 0.0
    c = grid[x_floor_plus_one, y_floor] if \
        (x_floor_plus_one is not None and y_floor is not None) else 0.0
    d = grid[x_floor_plus_one, y_floor_plus_one] if \
        (x_floor_plus_one is not None and y_floor_plus_one is not None) else 0.0

    #                                       [f(0,0) f(0,1)] [1-y]
    # Calculate Interp value with    [1-x x] [f(1,0) f(1,1)] [ y ]
    val_matrix = np.array([[a, b], [c, d]])

    return np.matmul(np.matmul(np.array([1 - x_p, x_p]), val_matrix), np.transpose(np.array([1 - y_p, y_p])))


def rebin_curved_to_flat_detector(args, proj_curved_helic):
    """ Rebin cylindrically curved detector projections to flat detector projections. Simultaneously, the central
    detector position (det_central_element) is shifted to the real geometric center of the curved detector.

    :param args: required geometry parameters
    :param proj_curved_helic: curved detector projections
    :return: rebinned detector projections
    """
    proj_flat_helic = np.zeros_like(proj_curved_helic, dtype=np.float32)

    # Origin in mm located at source position == focal center of detector [x=column, y=dsd-direction, z=row].
    p_0 = np.array([0., 0., 0.])

    for i_angle in tqdm.tqdm(range(proj_curved_helic.shape[0]), 'Rebin curved to flat detector'):
        for i_x_det in range(proj_curved_helic.shape[2]):
            for i_z_det in range(proj_curved_helic.shape[1]):
                # Calculate x and z coordinate of flat detector pixel. Y is given by dsd. Consider 0.5 pixel shift.
                x_det = (i_x_det - args.nu / 2) * args.du + 0.5 * args.du
                z_det = (i_z_det - args.nv / 2) * args.dv + 0.5 * args.dv
                p_on_flat_det = np.array([x_det, args.dsd, z_det])

                # Find the position of a ray (hitting a pixel on the virtual flat detector)
                # crossing the curved detector.
                p_on_curved_det = p_0 + (p_on_flat_det / np.linalg.norm(p_on_flat_det)) * args.dsd
                # Find the corresponding angle within the fan via trigonometry.
                phi_on_curved_det = np.arcsin(p_on_curved_det[0] / args.dsd)

                # Create 2D grid of curved detector with angular direction of step size dphi_curved and axial
                # direction v. The center of the virtual flat detector is chosen as the real geometric center of the
                # curved detector. Therefore, the det_central_element is used to shift the curved detector to the
                # correct position.
                dphi_curved = 2 * np.arctan(args.du / (2 * args.dsd))
                i_p_on_curved_det_polar = np.array([phi_on_curved_det / dphi_curved, p_on_curved_det[2] / args.dv]) + \
                                          np.array([args.nu - args.det_central_element[0], args.nv - args.det_central_element[1]])

                p_interp = interpolate_2d(grid=proj_curved_helic[i_angle, :, :],
                                          x=i_p_on_curved_det_polar[1],
                                          y=i_p_on_curved_det_polar[0])

                proj_flat_helic[i_angle, i_z_det, i_x_det] = p_interp

    return proj_flat_helic


def _rebin_curved_to_flat_detector_core(args, proj_curved_helic, i_angle):
    """ Core loops to rebin cylindrically curved detector projections to flat detector projections. Simultaneously,
    the central detector position (det_central_element) is shifted to the real geometric center of the curved detector.

    :param args: required geometry parameters
    :param proj_curved_helic: curved detector projections
    :param i_angle: index of rebinned projection for multiprocessing
    :return: rebinned detector projections
    """

    proj_flat_helic_i_angle = np.zeros_like(proj_curved_helic[i_angle], dtype=np.float32)

    # Origin in mm located at source position == focal center of detector [x=column, y=dsd-direction, z=row].
    p_0 = np.array([0., 0., 0.])

    for i_x_det in range(proj_curved_helic.shape[2]):
        for i_z_det in range(proj_curved_helic.shape[1]):
            # Calculate x and z coordinate of flat detector pixel. Y is given by dsd. Consider 0.5 pixel shift.
            x_det = (i_x_det - args.nu / 2) * args.du + 0.5 * args.du
            # dz_ffs = (args.ddo / args.dso) * args.dz[i_angle]
            z_det = (i_z_det - args.nv / 2) * args.dv + 0.5 * args.dv
            p_on_flat_det = np.array([x_det, args.dsd, z_det])

            # Find the position of a ray (hitting a pixel on the virtual flat detector)
            # crossing the curved detector.
            p_on_curved_det = p_0 + (p_on_flat_det / np.linalg.norm(p_on_flat_det)) * args.dsd
            # Find the corresponding angle within the fan via trigonometry.
            phi_on_curved_det = np.arcsin(p_on_curved_det[0] / args.dsd)

            # Create 2D grid of curved detector with angular direction of step size dphi_curved and axial
            # direction v. The center of the virtual flat detector is chosen as the real geometric center of the
            # curved detector. Therefore, the det_central_element is used to shift the curved detector to the
            # correct position.
            dphi_curved = 2 * np.arctan(args.du / (2 * args.dsd))
            i_p_on_curved_det_polar = np.array([phi_on_curved_det / dphi_curved, p_on_curved_det[2] / args.dv]) + \
                                      np.array([args.nu - args.det_central_element[0], args.nv - args.det_central_element[1]])

            p_interp = interpolate_2d(grid=proj_curved_helic[i_angle, :, :],
                                      x=i_p_on_curved_det_polar[1],
                                      y=i_p_on_curved_det_polar[0])

            proj_flat_helic_i_angle[i_z_det, i_x_det] = p_interp

    return proj_flat_helic_i_angle


def rebin_curved_to_flat_detector_multiprocessing(data, cols):
    """ Function to rebin curved detector data to a flat panel. Needs to be called when multiprocessing with joblib.

    :param data: tuple of (args, proj_curved_helic)
    :param cols: angular index to iterate over projections
    :return: rebinned detector projections
    """
    args, proj_curved_helic = data

    return _rebin_curved_to_flat_detector_core(args, proj_curved_helic, cols)


def rebin_helical_to_fan_beam_trajectory(args, proj_helic):
    """ Rebin projections acquired on a helical trajectory to full-scan (2pi) fan beam projections.

    :param args: required geometry parameters
    :param proj_helic: projections acquired on a helical trajectory
    :return: rebinned fan beam projections
    """
    distance = 0.5 * args.pitch  # Full scan. For short scan see Noo et al. "Single-slice rebinning ...".

    proj_rebinned = np.zeros((args.rotview, args.nu, args.nz_rebinned), dtype=np.float32)

    # Loop over view angles.
    for s_angle in tqdm.tqdm(range(args.rotview), 'Rebin helical to fan-beam geometry'):
        # Find all valid projections at s_angle.
        z_poses_valid = args.z_positions[s_angle::args.rotview]

        # Axial positions of resampled projections, starting at z_positions[0].
        z_poses_resampled = (np.arange(0, args.nz_rebinned, 1) * args.dv_rebinned) + args.z_positions[0]

        for i_proj in range(len(z_poses_valid)):
            # Calculate indices of lower and upper z limit for each valid cone beam source position.
            lower_lim = z_poses_valid[i_proj] - distance
            upper_lim = z_poses_valid[i_proj] + distance

            i_lower_lim = np.clip(int((lower_lim - args.z_positions[0]) / args.dv_rebinned),
                                  a_min=0, a_max=len(z_poses_resampled))
            i_upper_lim = np.clip(int(np.ceil((upper_lim - args.z_positions[0]) / args.dv_rebinned)),
                                  a_min=0, a_max=len(z_poses_resampled))

            # Loop over z slices.
            for i_z_resampled in range(i_lower_lim, i_upper_lim):
                # Axial distance between virtual and helical projection.
                deltaZ = z_poses_valid[i_proj] - z_poses_resampled[i_z_resampled]

                for i_u in range(args.nu):
                    # Analytically calculated detector position of the rebinned detector pixel that
                    # needs to be interpolated on the discrete scanner detector. Consider 0.5 pixel shift.
                    # Eq.(2) from Noo et al. "Single-slice rebinning method for helical cone-beam CT" (1999).
                    v_precise = deltaZ * \
                                ((((i_u - args.nu / 2 + 0.5) * args.du)**2 + args.dsd**2) /
                                 (args.dso * args.dsd))

                    # Array of v locations of the detector elements relative to the central v detector element.
                    # Consider 0.5 pixel shift.
                    v_det_elements = (np.arange(0, args.nv, 1) - args.nv / 2 + 0.5) * args.dv
                    v_det_values = proj_helic[s_angle + i_proj * args.rotview, :, i_u]

                    # Interpolated v_precise on the projection acquired on the helical trajectory.
                    v_interp = np.interp(x=v_precise, xp=v_det_elements, fp=v_det_values)

                    # Eq.(1) from Noo et al. "Single-slice rebinning method for helical cone-beam CT" (1999).
                    # Consider 0.5 pixel shift.
                    proj_rebinned[s_angle, i_u, i_z_resampled] = \
                        (np.sqrt(((i_u - args.nu / 2 + 0.5) * args.du)**2 + args.dsd**2) /
                         np.sqrt(((i_u - args.nu / 2 + 0.5) * args.du)**2 +
                                 v_precise**2 + args.dsd**2)) \
                        * v_interp

    return proj_rebinned
