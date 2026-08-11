"""
reconstruction_leveling.py

Corrects an OpenSfM reconstruction.json's global orientation so its Z axis
actually points along true gravity, instead of whatever orientation the
bundle adjustment happened to converge on. Without RTK, that orientation is
only as good as GPS + feature matching can constrain it, and can end up
tilted by a real, physical angle relative to true level - which then shows
up as sloped floors/terrain no amount of downstream processing can fix,
since it's baked into scene_dense.ply itself.

Two independent methods, cross-checked against each other when both are
available:

1. DJI gimbal attitude (from XMP metadata embedded in each photo) - direct
   orientation measurements from the drone's own IMU, independent of GPS
   entirely.

2. GPS camera-position alignment (Umeyama similarity fit between each shot's
   reconstructed camera center and its EXIF GPS position) - universal
   fallback, since plain GPS lat/lon/altitude is standard EXIF on virtually
   any drone, DJI or not. Noisier than the DJI method (derived from noisy
   position data rather than direct orientation), but averaged over many
   shots it's normally precise enough to correct a real, multi-degree tilt.

TILT AND YAW ARE TRUSTED SEPARATELY, NOT AS ONE MONOLITHIC ROTATION - this
was a real, confirmed bug: DJI's GimbalYawDegree is referenced to the
aircraft's onboard COMPASS, which reads MAGNETIC north, not true north.
Nothing in this file corrected for magnetic declination, so on a real
project this method's "gravity correction" was silently also rotating the
whole reconstruction by ~24 degrees of pure heading error, confirmed both by
this file's own DJI-vs-GPS disagreement warning (24.1 vs 12.5 degrees) AND
independently by comparing final building positions against ground-truth
parcel data. The gimbal's IMU has no comparable reason to be biased for
TILT (accelerometers measure gravity directly, not compass heading) - only
YAW is the physically suspect component. So: every rotation this file
produces is decomposed into a swing (tilt: rotation about a horizontal
axis, bringing gravity into alignment) and a twist (yaw: rotation about the
vertical axis, i.e. heading) about the true-vertical axis. DJI's swing is
still trusted directly. DJI's twist is only applied when it's corroborated
by the GPS method's own twist (which isn't subject to a magnetic bias,
only ordinary position noise) - agreeing within YAW_AGREEMENT_DEG. When
they disagree by more than that, yaw goes uncorrected rather than risk
repeating this exact failure, and a specific, greppable warning is printed
identifying it as a probable compass/declination issue (as opposed to the
old, vaguer "these disagree, trusting DJI anyway" warning, which fired for
exactly this failure and wasn't specific enough for anyone to act on it).

Both methods, and the swing-twist decomposition, were validated against
synthetic ground truth before being wired into the real pipeline - see the
project conversation history for the test cases (nadir/level/heading
sanity checks for the DJI convention, noise-tolerance checks for the
Umeyama fit, and exact tilt/yaw recovery checks for the decomposition,
including a direct reproduction of the real 24.113-degree case).
"""
import os
import re
import json
import numpy as np

# --- Fixed, known coordinate-frame conventions (not fitted - these are
# standard, documented relationships, unlike the actual tilt we're solving for) ---

# DJI body frame (X=forward, Y=right, Z=down) -> camera optical frame
# (X=right, Y=down, Z=forward/into-scene, the standard computer-vision convention).
_BODY_TO_OPTICAL = np.array([[0., 1., 0.],
                              [0., 0., 1.],
                              [1., 0., 0.]])

# NED (X=North, Y=East, Z=Down) -> ENU (X=East, Y=North, Z=Up), OpenSfM's
# typical local topocentric convention.
_NED_TO_ENU = np.array([[0., 1., 0.],
                         [1., 0., 0.],
                         [0., 0., -1.]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def rodrigues_to_matrix(rvec):
    """Rodrigues rotation vector -> 3x3 rotation matrix, pure NumPy (matches
    cv2.Rodrigues to floating-point precision - validated separately). Avoids
    an OpenCV dependency in what's otherwise a lightweight orchestration
    script with no other image-processing needs."""
    rvec = np.asarray(rvec, dtype=float).flatten()
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    axis = rvec / theta
    K = np.array([[0., -axis[2], axis[1]],
                  [axis[2], 0., -axis[0]],
                  [-axis[1], axis[0], 0.]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def matrix_to_rodrigues(R):
    """3x3 rotation matrix -> Rodrigues rotation vector, pure NumPy."""
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Near 180 degrees the standard formula divides by ~0 - extract the
        # axis from the symmetric part instead (rare in practice for camera
        # orientations, but handled for correctness).
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(axis))
        for j in range(3):
            if j != i and axis[i] > 1e-8:
                axis[j] = A[i, j] / axis[i]
        axis = axis / np.linalg.norm(axis)
        return axis * theta
    rx = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
    return np.array([rx, ry, rz]) * theta


# --- Swing-twist decomposition ---
# Splits a rotation into a "twist" (rotation about a chosen axis) and a
# "swing" (the remainder, whose own axis is perpendicular to the twist
# axis) - standard technique from robotics/graphics. Used here to separate
# a gravity-correction rotation into its YAW component (twist about true
# vertical - i.e. heading, the part a magnetically-biased compass can get
# wrong) and its TILT component (swing about a horizontal axis - i.e.
# gravity alignment, which an IMU has no comparable reason to get wrong).
# Implemented via quaternions rather than matrices directly, since the
# projection step that isolates the twist has a clean, standard closed
# form in quaternion space. Validated against known synthetic tilt/yaw
# combinations, including an exact reproduction of the real 8-degree-tilt
# plus 24.113-degree-yaw-bias case this was built to fix.
_UP_AXIS = np.array([0., 0., 1.])


def _matrix_to_quaternion(R):
    """3x3 rotation matrix -> quaternion (w,x,y,z). Numerically stable
    (Shepperd's method: picks whichever of 4 equivalent formulas avoids
    dividing by something close to zero, based on the matrix trace/diagonal)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def swing_twist_decompose(R, twist_axis=_UP_AXIS):
    """Decomposes rotation R into (R_swing, R_twist) such that
    R = R_swing @ R_twist, R_twist is a pure rotation about `twist_axis`,
    and R_swing's own rotation axis is perpendicular to `twist_axis`.

    R is expressed in the same (world/output) basis `twist_axis` is given
    in - apply_correction_to_reconstruction applies R via left-
    multiplication (R @ coords), so this decomposition is with respect to
    the TARGET frame's vertical, not whatever frame the reconstruction
    started in (which may itself be tilted - that tilt is exactly what
    R_swing corrects)."""
    twist_axis = twist_axis / np.linalg.norm(twist_axis)
    q = _matrix_to_quaternion(R)
    w, vec = q[0], q[1:]
    proj = np.dot(vec, twist_axis) * twist_axis
    q_twist = np.array([w, *proj])
    norm = np.linalg.norm(q_twist)
    q_twist = np.array([1., 0., 0., 0.]) if norm < 1e-9 else q_twist / norm
    q_swing = _quaternion_multiply(q, _quaternion_conjugate(q_twist))
    return _quaternion_to_matrix(q_swing), _quaternion_to_matrix(q_twist)


def camera_center(rotation_rvec, translation):
    """OpenSfM shots store world-to-camera rotation (Rodrigues) + translation,
    i.e. X_cam = R @ X_world + t. Camera center in world coords: C = -R^T @ t."""
    R = rodrigues_to_matrix(rotation_rvec)
    t = np.asarray(translation, dtype=float)
    return -R.T @ t


def dji_gimbal_to_enu_rotation(yaw_deg, pitch_deg, roll_deg):
    """Converts DJI GimbalYaw/Pitch/RollDegree (already gravity/north
    referenced - the gimbal is independently IMU-stabilized against aircraft
    movement) into a camera-optical-to-ENU-world rotation matrix.

    Sanity-checked against known special cases: pitch=-90 (nadir) gives an
    optical axis pointing straight down; yaw=0/90 with pitch=0 give due
    North/East, level."""
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    r_body_to_ned = _rz(yaw) @ _ry(pitch) @ _rx(roll)
    r_optical_to_ned = r_body_to_ned @ _BODY_TO_OPTICAL.T
    return _NED_TO_ENU @ r_optical_to_ned


def read_dji_gimbal_attitude(image_path):
    """Returns (yaw, pitch, roll) in degrees from a DJI image's embedded XMP
    metadata, or None if the tags aren't present (non-DJI image, or DJI image
    without gimbal telemetry)."""
    try:
        with open(image_path, 'rb') as f:
            text = f.read().decode('latin-1', errors='ignore')
    except OSError:
        return None

    def _tag(name):
        m = re.search(rf'drone-dji:{name}="([+-]?[0-9.]+)"', text)
        return float(m.group(1)) if m else None

    yaw, pitch, roll = _tag('GimbalYawDegree'), _tag('GimbalPitchDegree'), _tag('GimbalRollDegree')
    if yaw is None or pitch is None or roll is None:
        return None
    return yaw, pitch, roll


def read_gps_latlonalt(image_path):
    """Standard EXIF GPS lat/lon/altitude - present on virtually any drone
    photo regardless of manufacturer, unlike gimbal telemetry."""
    from PIL import Image
    from PIL.ExifTags import GPSTAGS, TAGS
    try:
        exif = Image.open(image_path)._getexif()
        if not exif:
            return None
        gps = {GPSTAGS.get(t, t): v for t, v in exif[list(TAGS.keys())[list(TAGS.values()).index('GPSInfo')]].items()}
        def to_d(v): return float(v[0]) + float(v[1]) / 60.0 + float(v[2]) / 3600.0
        lat, lon = to_d(gps['GPSLatitude']), to_d(gps['GPSLongitude'])
        if gps.get('GPSLatitudeRef') != 'N':
            lat = -lat
        if gps.get('GPSLongitudeRef') != 'E':
            lon = -lon
        alt = float(gps['GPSAltitude']) if 'GPSAltitude' in gps else 0.0
        return lat, lon, alt
    except Exception:
        return None


def average_rotations(rotations):
    """Approximate average of a list of 3x3 rotation matrices: average the
    matrices elementwise, then re-orthonormalize via SVD. Standard, simple
    approximation for rotations that are all reasonably close together (which
    they should be here - they're all estimates of the same single global
    misalignment)."""
    mean_mat = np.mean(rotations, axis=0)
    U, _, Vt = np.linalg.svd(mean_mat)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_angle_deg(R):
    """Angle of a rotation matrix, in degrees."""
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def umeyama_alignment(src, dst):
    """Least-squares similarity transform (rotation R, scale s, translation t)
    minimizing sum ||dst_i - (s*R@src_i + t)||^2. Umeyama, IEEE TPAMI 1991.
    Returns (R, s, t)."""
    n, m = src.shape
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_src
    t = dst_mean - s * R @ src_mean
    return R, s, t


def compute_correction_from_dji(shots, images_dir):
    """Compares each shot's OpenSfM-recovered orientation to its DJI gimbal
    telemetry, and returns (R_correct, n_used) - the single rotation that
    best explains the discrepancy across all shots, or (None, 0) if no DJI
    telemetry was found."""
    diffs = []
    for filename, shot in shots.items():
        attitude = read_dji_gimbal_attitude(os.path.join(images_dir, filename))
        if attitude is None:
            continue
        yaw, pitch, roll = attitude
        r_cam_to_world_dji = dji_gimbal_to_enu_rotation(yaw, pitch, roll)
        r_wc_opensfm = rodrigues_to_matrix(shot['rotation'])
        r_cam_to_world_opensfm = r_wc_opensfm.T
        # world_opensfm = R_tilt @ world_true  =>  r_cam_to_world_opensfm = R_tilt @ r_cam_to_world_dji.
        # We need R_correct = R_tilt^-1 (to map opensfm-frame data back to true/gravity frame),
        # i.e. R_correct = r_cam_to_world_dji @ r_cam_to_world_opensfm.T (note the order - this is
        # R_tilt^T, not R_tilt; apply_correction_to_reconstruction expects the former).
        diffs.append(r_cam_to_world_dji @ r_cam_to_world_opensfm.T)

    if len(diffs) < 3:
        return None, len(diffs)
    return average_rotations(diffs), len(diffs)


def compute_correction_from_gps(shots, images_dir, reference_lla, epsg_code):
    """Aligns reconstructed camera centers to their EXIF GPS positions via a
    similarity transform, and returns (R_correct, n_used, residual_m) - or
    (None, 0, None) if not enough GPS-tagged shots were found."""
    from pyproj import Transformer
    trans = Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
    ref_x, ref_y = trans.transform(reference_lla['longitude'], reference_lla['latitude'])
    ref_alt = reference_lla.get('altitude', 0.0)

    recon_pts, gps_pts = [], []
    for filename, shot in shots.items():
        latlonalt = read_gps_latlonalt(os.path.join(images_dir, filename))
        if latlonalt is None:
            continue
        lat, lon, alt = latlonalt
        x, y = trans.transform(lon, lat)
        gps_pts.append([x - ref_x, y - ref_y, alt - ref_alt])
        recon_pts.append(camera_center(shot['rotation'], shot['translation']))

    if len(recon_pts) < 3:
        return None, len(recon_pts), None

    recon_pts, gps_pts = np.array(recon_pts), np.array(gps_pts)
    R, s, t = umeyama_alignment(recon_pts, gps_pts)
    residual = np.sqrt(np.mean(np.sum((gps_pts - (s * (R @ recon_pts.T).T + t)) ** 2, axis=1)))
    return R, len(recon_pts), residual


def apply_correction_to_reconstruction(reconstruction, R_correct):
    """Rotates every 3D point and every shot's camera rotation by R_correct,
    about the origin. Shot translations are unchanged - for a pure rotation
    of the world about the origin, only each camera's rotation component
    needs updating (see project notes for the derivation)."""
    for point in reconstruction.get('points', {}).values():
        coords = np.asarray(point['coordinates'], dtype=float)
        point['coordinates'] = (R_correct @ coords).tolist()

    for shot in reconstruction.get('shots', {}).values():
        R_wc = rodrigues_to_matrix(shot['rotation'])
        R_wc_new = R_wc @ R_correct.T
        rvec_new = matrix_to_rodrigues(R_wc_new)
        shot['rotation'] = rvec_new.flatten().tolist()


def level_reconstruction(project_path, epsg_code="EPSG:32612", tilt_warn_deg=5.0, yaw_agreement_deg=5.0):
    """Loads reconstruction.json, computes tilt and yaw corrections
    SEPARATELY (DJI gimbal telemetry primary for tilt; DJI for yaw only
    when GPS corroborates it, GPS alone otherwise - see the module
    docstring for why), applies them, and writes the corrected
    reconstruction.json back out. Returns True if any correction was
    applied.

    tilt_warn_deg: how much DJI/GPS tilt estimates can disagree before
    printing a warning (tilt is always applied from whichever method is
    available regardless - this disagreement isn't the known bias, so it's
    informational, not a reason to withhold the correction).

    yaw_agreement_deg: how closely DJI/GPS yaw (twist) estimates must agree
    before DJI's yaw is trusted. This one DOES gate the correction: this is
    the specific measurement with a known, confirmed failure mode (DJI's
    compass reads magnetic, not true, north), so disagreement beyond this
    threshold means yaw goes uncorrected rather than risk repeating it."""
    recon_path = os.path.join(project_path, 'reconstruction.json')
    images_dir = os.path.join(project_path, 'images')

    if not os.path.exists(recon_path):
        print("      [!] No reconstruction.json found - skipping gravity-leveling correction")
        return False

    with open(recon_path, 'r') as f:
        reconstruction = json.load(f)
    recon = reconstruction[0]
    shots = recon.get('shots', {})
    reference_lla = recon.get('reference_lla')

    print(f"      -> Checking {len(shots)} shots for DJI gimbal telemetry...")
    R_dji, n_dji = compute_correction_from_dji(shots, images_dir)

    R_gps, n_gps, residual_gps = (None, 0, None)
    if reference_lla is not None:
        print(f"      -> Checking shots for GPS EXIF (fallback/cross-check method)...")
        R_gps, n_gps, residual_gps = compute_correction_from_gps(shots, images_dir, reference_lla, epsg_code=epsg_code)

    if R_dji is not None:
        print(f"      -> DJI method: {n_dji}/{len(shots)} shots had usable gimbal telemetry, "
              f"correction angle {rotation_angle_deg(R_dji):.3f} degrees")
    else:
        print(f"      -> DJI method unavailable ({n_dji} shots had gimbal telemetry, need >= 3)")

    if R_gps is not None:
        print(f"      -> GPS method: {n_gps}/{len(shots)} shots had usable GPS EXIF, "
              f"correction angle {rotation_angle_deg(R_gps):.3f} degrees, "
              f"fit residual {residual_gps:.2f}m")
    else:
        print(f"      -> GPS method unavailable ({n_gps} shots had usable GPS EXIF, need >= 3)")

    # Decompose each available correction into TILT (swing about a
    # horizontal axis - gravity alignment) and YAW (twist about true
    # vertical - heading) - see the module docstring for why these get
    # different trust levels below.
    swing_dji, twist_dji = swing_twist_decompose(R_dji) if R_dji is not None else (None, None)
    swing_gps, twist_gps = swing_twist_decompose(R_gps) if R_gps is not None else (None, None)

    if swing_dji is not None:
        print(f"      -> DJI decomposed: tilt={rotation_angle_deg(swing_dji):.3f} deg, "
              f"yaw={rotation_angle_deg(twist_dji):.3f} deg")
    if swing_gps is not None:
        print(f"      -> GPS decomposed: tilt={rotation_angle_deg(swing_gps):.3f} deg, "
              f"yaw={rotation_angle_deg(twist_gps):.3f} deg")

    tilt_agreement = yaw_agreement = None
    if swing_dji is not None and swing_gps is not None:
        tilt_agreement = rotation_angle_deg(swing_dji.T @ swing_gps)
        yaw_agreement = rotation_angle_deg(twist_dji.T @ twist_gps)
        print(f"      -> Tilt agreement (DJI vs GPS): {tilt_agreement:.3f} degrees")
        print(f"      -> Yaw agreement  (DJI vs GPS): {yaw_agreement:.3f} degrees")
        if tilt_agreement > tilt_warn_deg:
            print(f"      [!] WARNING: tilt estimates disagree by more than {tilt_warn_deg} degrees - "
                  f"this is NOT the known magnetic-yaw issue (that only affects heading, not gravity "
                  f"alignment) - worth checking GPS data quality and DJI gimbal telemetry for this run.")

    # Tilt: always trust DJI's swing when available - an IMU/accelerometer
    # has no known comparable reason to be biased for gravity alignment.
    # GPS's swing is the fallback when DJI isn't available.
    if swing_dji is not None:
        tilt_correct, tilt_method = swing_dji, "DJI gimbal telemetry"
    elif swing_gps is not None:
        tilt_correct, tilt_method = swing_gps, "GPS position alignment"
    else:
        tilt_correct, tilt_method = None, None

    # Yaw: only trust DJI's twist when GPS corroborates it within
    # yaw_agreement_deg - this is the actual fix for the confirmed bug.
    # GPS's twist alone (DJI unavailable) is still trusted on its own,
    # since it isn't subject to a comparable systematic bias, only ordinary
    # position noise. An UNCORROBORATED DJI twist (GPS unavailable) is
    # treated the same as a disagreeing one and skipped - the whole point
    # is that DJI's yaw specifically needs a second opinion, not just the
    # absence of a contradiction.
    if twist_dji is not None and twist_gps is not None and yaw_agreement <= yaw_agreement_deg:
        yaw_correct = twist_dji
        yaw_method = f"DJI gimbal telemetry (corroborated by GPS, {yaw_agreement:.2f} deg apart)"
    elif twist_dji is not None and twist_gps is not None:
        yaw_correct, yaw_method = None, None
        print(f"      [!] WARNING: DJI and GPS YAW estimates disagree by {yaw_agreement:.3f} degrees "
              f"(tilt estimates agree within {tilt_agreement:.3f} degrees) - this specific pattern "
              f"(tilt agrees, yaw doesn't) is the known failure mode: DJI's gimbal yaw is referenced "
              f"to MAGNETIC north, and nothing here corrects for local declination. No yaw correction "
              f"applied this run. This dataset's absolute orientation - not its shape, tilt, or "
              f"internal measurements - may still be off by tens of degrees; verify against a known "
              f"heading if you need accurate lat/lon.")
    elif twist_gps is not None:
        yaw_correct, yaw_method = twist_gps, "GPS position alignment (uncorroborated - DJI unavailable)"
    elif twist_dji is not None:
        yaw_correct, yaw_method = None, None
        print("      [!] DJI yaw available but uncorroborated (no usable GPS EXIF to cross-check) - "
              "skipping yaw correction rather than risk applying an unverified magnetic-compass bias. "
              "Tilt correction is still applied from DJI.")
    else:
        yaw_correct, yaw_method = None, None

    if tilt_correct is None and yaw_correct is None:
        print("      [!] No usable tilt or yaw correction from either method - no leveling "
              "correction applied. scene_dense.ply may retain whatever tilt/heading the "
              "reconstruction converged on.")
        return False

    if tilt_correct is not None:
        print(f"      -> Applying {rotation_angle_deg(tilt_correct):.3f} degree TILT correction from {tilt_method}")
    else:
        print("      -> No tilt correction applied (no usable method)")
    if yaw_correct is not None:
        print(f"      -> Applying {rotation_angle_deg(yaw_correct):.3f} degree YAW correction from {yaw_method}")
    else:
        print("      -> No yaw correction applied - see warnings above if this is unexpected")

    R_correct = (tilt_correct if tilt_correct is not None else np.eye(3)) @ \
                (yaw_correct if yaw_correct is not None else np.eye(3))
    apply_correction_to_reconstruction(recon, R_correct)

    with open(recon_path, 'w') as f:
        json.dump(reconstruction, f, indent=4)
    print(f"      -> Corrected reconstruction.json written to: {recon_path}")
    return True
