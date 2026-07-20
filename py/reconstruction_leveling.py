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
   entirely. Primary method when the drone is DJI.

2. GPS camera-position alignment (Umeyama similarity fit between each shot's
   reconstructed camera center and its EXIF GPS position) - universal
   fallback, since plain GPS lat/lon/altitude is standard EXIF on virtually
   any drone, DJI or not. Noisier than the DJI method (derived from noisy
   position data rather than direct orientation), but averaged over many
   shots it's normally precise enough to correct a real, multi-degree tilt.

Both methods were validated against synthetic ground truth before being
wired into the real pipeline - see the project conversation history for the
test cases (nadir/level/heading sanity checks for the DJI convention, and
noise-tolerance checks for the Umeyama fit).
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


def level_reconstruction(project_path, epsg_code="EPSG:32612", agreement_warn_deg=2.0):
    """Loads reconstruction.json, computes a gravity-alignment correction
    (DJI gimbal telemetry primary, GPS-position alignment as a universal
    fallback and cross-check), applies it, and writes the corrected
    reconstruction.json back out. Returns True if a correction was applied."""
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

    if R_dji is not None and R_gps is not None:
        agreement = rotation_angle_deg(R_dji.T @ R_gps)
        print(f"      -> DJI vs GPS methods disagree by {agreement:.3f} degrees")
        if agreement > agreement_warn_deg:
            print(f"      [!] WARNING: disagreement exceeds {agreement_warn_deg} degrees - "
                  f"one of these is likely wrong. Applying the DJI correction (primary), but "
                  f"treat the result with caution until you've checked it against something "
                  f"you know is truly level in the scene (e.g. a visible water surface).")

    R_correct = R_dji if R_dji is not None else R_gps
    method = "DJI gimbal telemetry" if R_dji is not None else "GPS position alignment"

    if R_correct is None:
        print("      [!] Neither DJI telemetry nor GPS EXIF was usable - no leveling "
              "correction applied. scene_dense.ply may retain whatever tilt the "
              "reconstruction converged on.")
        return False

    print(f"      -> Applying {rotation_angle_deg(R_correct):.3f} degree correction from {method}")
    apply_correction_to_reconstruction(recon, R_correct)

    with open(recon_path, 'w') as f:
        json.dump(reconstruction, f, indent=4)
    print(f"      -> Corrected reconstruction.json written to: {recon_path}")
    return True
