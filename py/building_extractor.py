import os

# --- SYSTEM CONFIGURATION ---
# Must be set before numpy/scipy/sklearn/open3d are imported below - their
# BLAS thread pools initialize at import time, so setting these after import
# (where they used to live) has no effect. Left unset, each of the up to 32
# worker processes in the extraction pool could spin up its own multi-threaded
# BLAS, oversubscribing a 12-16 core allocation many times over.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import numpy as np
import open3d as o3d
import csv
import sys
import time
import shutil
import glob
import cv2
import argparse
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import Delaunay, KDTree
from scipy.ndimage import binary_erosion, label, grey_opening, gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pyproj import Transformer
import concurrent.futures
from resource_monitor import detect_cpu_count, MemoryMonitor

# Line-buffer stdout: when redirected to a file (e.g. a SLURM .out log),
# Python block-buffers by default and only flushes at exit. If the job gets
# killed (timeout, OOM) instead of exiting cleanly, everything this script
# printed - peak memory, step progress, resource sizing info - is lost.
sys.stdout.reconfigure(line_buffering=True)

'''
extract_buildings_no_floor.py

NOTE: despite the filename, this version DOES synthesize a floor again - it
started as a no-floor variant, then went back to using the diagnostic ground
model (the same ground_surface.ply you can already inspect) as the floor
source, instead of extract_buildings.py's min-based "cut to the lowest point"
approach, which was landing way too low. The floor here is the MEDIAN of the
ground model's samples across the footprint (a typical/representative ground
elevation under the building), not the minimum, and not a per-cell average of
the raw natural-terrain surface - still one flat value per house. Consider
renaming this file once you're happy with the result.

Same pipeline as extract_buildings.py otherwise (same leveling,
LocalGroundModel, vegetation removal, erosion/fragment-reconciliation
isolation logic). Height_ft/Volume_cuft are computed relative to this
ground-model-derived floor, same as extract_buildings.py.

Run using: python extract_buildings_no_floor.py data/900EBlock
'''

EPSG_CODE = "EPSG:32612"
NUM_CORES, _CORE_SOURCE = detect_cpu_count()

# --- 1. LOCAL GROUND MODEL ---
class LocalGroundModel:
    """Builds a locally-varying ground elevation surface ('contour map')
    directly from the point cloud: per-cell minimum elevation, then a
    morphological opening wide enough to punch through building/canopy cover
    but not real terrain slope. Used here only for above-ground filtering,
    vegetation removal and the roof-seed height threshold - NOT for
    synthesizing a floor (see module docstring)."""

    def __init__(self, pts, cell_size=2.0, opening_span=20.0):
        self.cell_size = cell_size
        self.opening_span = opening_span

        xy_min = pts[:, :2].min(axis=0) - cell_size
        xy_max = pts[:, :2].max(axis=0) + cell_size
        nx = int(np.ceil((xy_max[0] - xy_min[0]) / cell_size)) + 1
        ny = int(np.ceil((xy_max[1] - xy_min[1]) / cell_size)) + 1

        ix = np.clip(((pts[:, 0] - xy_min[0]) / cell_size).astype(int), 0, nx - 1)
        iy = np.clip(((pts[:, 1] - xy_min[1]) / cell_size).astype(int), 0, ny - 1)
        flat_idx = ix * ny + iy

        grid_min = np.full(nx * ny, np.inf)
        np.minimum.at(grid_min, flat_idx, pts[:, 2])
        occupied = np.isfinite(grid_min).reshape(nx, ny)
        grid_min = grid_min.reshape(nx, ny)

        empty_cells = int((~occupied).sum())
        if empty_cells:
            _, nearest_idx = distance_transform_edt(~occupied, return_distances=True, return_indices=True)
            grid_min = grid_min[tuple(nearest_idx)]

        # Punch through anything narrower than `opening_span` (roofs, tree
        # canopies act as false-high "ground") while leaving broader real
        # terrain slope/hills untouched since they're much wider than a building.
        k = max(1, int(round(opening_span / cell_size)))
        ground_grid = grey_opening(grid_min, size=(k, k))
        ground_grid = gaussian_filter(ground_grid, sigma=max(0.5, k / 6.0))

        self.xs = xy_min[0] + (np.arange(nx) + 0.5) * cell_size
        self.ys = xy_min[1] + (np.arange(ny) + 0.5) * cell_size
        self.grid = ground_grid
        self.interp = RegularGridInterpolator(
            (self.xs, self.ys), ground_grid, bounds_error=False, fill_value=None
        )

        print(f"      -> Ground grid: {nx}x{ny} cells ({cell_size}m each), "
              f"{empty_cells} empty cells filled from neighbors")
        print(f"      -> Opening kernel: {k}x{k} cells (~{k*cell_size:.1f}m span)")
        print(f"      -> Ground elevation range across site: "
              f"{ground_grid.min():.2f}m to {ground_grid.max():.2f}m "
              f"(relief: {ground_grid.max()-ground_grid.min():.2f}m)")

    def get_z(self, x, y):
        """Returns the true local ground elevation for arrays of X, Y coordinates."""
        xq = np.clip(x, self.xs[0], self.xs[-1])
        yq = np.clip(y, self.ys[0], self.ys[-1])
        return self.interp(np.stack([xq, yq], axis=-1))

    def write_diagnostic_ply(self, out_path, off_xy, rot):
        """Writes the ground surface as a colored point cloud (blue=low,
        red=high), transformed back into the original scene coordinates so it
        can be loaded next to scene_dense.ply in a viewer for a quick sanity
        check without waiting on the full extraction to finish."""
        xx, yy = np.meshgrid(self.xs, self.ys, indexing='ij')
        pts = np.stack([xx.ravel(), yy.ravel(), self.grid.ravel()], axis=1)

        z = pts[:, 2]
        z_range = max(1e-6, z.max() - z.min())
        z_norm = (z - z.min()) / z_range
        colors = np.stack([z_norm, np.zeros_like(z_norm), 1.0 - z_norm], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.translate((off_xy[0], off_xy[1], 0))
        pcd.rotate(rot.T, center=(0, 0, 0))
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"      -> Ground surface diagnostic written to: {out_path}")

# --- 2. AUTOMATION & EXTRACTION HELPERS ---
def auto_detect_offsets(project_dir):
    json_path = os.path.join(project_dir, 'reconstruction.json')
    if not os.path.exists(json_path):
        print(f"      [!] No reconstruction.json found at {json_path}; using (0.0, 0.0) global offset")
        return 0.0, 0.0
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ref = data[0].get('reference_lla')
        trans = Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True)
        offset = trans.transform(ref['longitude'], ref['latitude'])
        print(f"      -> Loaded reference LLA from {json_path}")
        print(f"      -> Global UTM offset: X={offset[0]:.3f}, Y={offset[1]:.3f}")
        return offset
    except Exception as e:
        print(f"      [!] Failed to read {json_path} ({e}); using (0.0, 0.0) global offset")
        return 0.0, 0.0

def build_spatial_image_index(image_dir, transformer):
    paths, coords = [], []
    if not image_dir or not os.path.exists(image_dir):
        print(f"      [!] Image directory not found: {image_dir} (no source photos will be attached)")
        return None, None
    files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.JPG"))
    for p in files:
        try:
            exif = Image.open(p)._getexif()
            if not exif:
                continue
            gps = {GPSTAGS.get(t, t): v for t, v in exif[list(TAGS.keys())[list(TAGS.values()).index('GPSInfo')]].items()}
            def to_d(v): return float(v[0]) + float(v[1])/60.0 + float(v[2])/3600.0
            lat, lon = to_d(gps['GPSLatitude']), to_d(gps['GPSLongitude'])
            if gps['GPSLatitudeRef'] != 'N':
                lat = -lat
            if gps['GPSLongitudeRef'] != 'E':
                lon = -lon
            paths.append(p)
            coords.append(transformer.transform(lon, lat))
        except:
            continue
    print(f"      -> Indexed {len(paths)}/{len(files)} images with usable GPS EXIF from {image_dir}")
    return (paths, KDTree(np.array(coords))) if coords else (None, None)

def repair_openmvs_ply_colors(ply_path):
    with open(ply_path, 'rb') as f:
        header_chunk = f.read(8192)
    if b'diffuse_red' not in header_chunk:
        return ply_path

    header_end_idx = header_chunk.find(b'end_header')
    if header_end_idx == -1:
        # Header longer than our peek window (very unusual) - read more.
        with open(ply_path, 'rb') as f:
            header_chunk = f.read(1 << 20)
        header_end_idx = header_chunk.find(b'end_header')
    newline_idx = header_chunk.find(b'\n', header_end_idx)
    header_len = newline_idx + 1
    header = header_chunk[:header_len]

    # Pad each replacement name to the exact byte length of the name it's
    # replacing (ljust, not hand-counted spaces) - a previous version of this
    # function hand-counted the padding and was 1-2 bytes short on 2 of the 3
    # names. That was harmless under a whole-file rewrite (nothing needs a
    # fixed offset there), but this function now patches a copy's header
    # bytes in place, which requires the header's length to stay exactly the
    # same - constructing the padding this way makes that guaranteed rather
    # than manually counted.
    for old_name, new_name in [(b'diffuse_red', b'red'), (b'diffuse_green', b'green'), (b'diffuse_blue', b'blue')]:
        header = header.replace(b'property uchar ' + old_name,
                                 b'property uchar ' + new_name.ljust(len(old_name)))
    assert len(header) == header_len, "header patch changed length - refusing to risk corrupting the point cloud"

    # Copy the file at the OS level (fast, low memory) instead of reading the
    # whole multi-GB point cloud into RAM just to rewrite ~200 bytes, then
    # patch the header of the copy in place.
    fixed_path = ply_path.replace('.ply', '_color_fixed.ply')
    shutil.copy2(ply_path, fixed_path)
    with open(fixed_path, 'r+b') as f:
        f.seek(0)
        f.write(header)

    print(f"      -> OpenMVS 'diffuse_*' color header detected; repaired copy written to: {fixed_path}")
    return fixed_path

def calculate_alpha_shape(points_2d, alpha=1.2):
    if len(points_2d) < 4:
        return None
    try:
        tri = Delaunay(points_2d)
        edges = np.array([points_2d[tri.simplices[:, [0, 1]]], points_2d[tri.simplices[:, [1, 2]]], points_2d[tri.simplices[:, [2, 0]]]])
        lengths = np.sqrt(np.sum((edges[:, :, 0, :] - edges[:, :, 1, :])**2, axis=2))
        circum_r = lengths[0,:]*lengths[1,:]*lengths[2,:] / (np.sqrt((lengths[0,:]+lengths[1,:]+lengths[2,:])*(-lengths[0,:]+lengths[1,:]+lengths[2,:])*(lengths[0,:]-lengths[1,:]+lengths[2,:])*(lengths[0,:]+lengths[1,:]-lengths[2,:])))
        valid = tri.simplices[np.nan_to_num(circum_r) < alpha]
        if len(valid) == 0:
            return None
        hull_pts = np.concatenate([points_2d, np.zeros((len(points_2d), 1))], axis=1)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hull_pts), o3d.utility.Vector3iVector(valid))
        return o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.3), valid, hull_pts
    except:
        return None

# "Is this candidate substantial enough to bother with" - expressed in real
# sqft, not raw point count. A fixed point count means a different physical
# size depending on point density, which silently shifts with e.g.
# --depthmap-resolution - the same "100 points" that meant ~36 sqft (a shed)
# at ~30 pts/m^2 means ~2.7 sqft (a doormat) at ~400 pts/m^2. This is a low,
# "clearly not just noise" bar - the real building-size decision happens
# later (the Ratio_Area_to_Height KMeans threshold in step 9).
MIN_CANDIDATE_AREA_SQFT = 20.0

def occupied_area_sqft(xy, cell_size=0.25):
    """Physical area (sqft) actually covered by these 2D points, counting
    unique occupied cells at `cell_size` resolution - not raw point count.
    Same technique the fragment-reconciliation merge logic already uses for
    its own (already density-independent) area_sqft."""
    if len(xy) == 0:
        return 0.0
    origin = xy.min(axis=0)
    cells = np.floor((xy - origin) / cell_size).astype(np.int64)
    return len(np.unique(cells, axis=0)) * (cell_size ** 2) * 10.764

def apply_house_cookie_cutter(ag_pts, ag_colors, footprint_data, ground_model):
    """Crops the point cloud to one building's footprint and builds its
    synthetic floor from the diagnostic LocalGroundModel: the MEDIAN of the
    ground model's samples across the footprint (a typical/representative
    ground elevation under the building - the same surface ground_surface.ply
    shows you), used as a single flat elevation for the whole floor. This
    replaces extract_buildings.py's min-based "cut to the lowest point"
    approach, which was landing way too low on sloped sites."""
    voxel_grid, _, _ = footprint_data
    contain = voxel_grid.check_if_included(o3d.utility.Vector3dVector(ag_pts * [1., 1., 0.]))
    f_pts, f_cols = ag_pts[contain], ag_colors[contain]
    if occupied_area_sqft(f_pts[:, :2]) < MIN_CANDIDATE_AREA_SQFT:
        return None

    mins, maxs = voxel_grid.get_min_bound(), voxel_grid.get_max_bound()
    res, cell_a = 0.6, 0.36
    vol, area = 0.0, 0.0

    # Generate the 2D footprint grid for intersection math
    xs, ys = np.arange(mins[0], maxs[0], res), np.arange(mins[1], maxs[1], res)
    grid = np.array([[x, y, 0.0] for x in xs for y in ys])

    if len(grid) > 0:
        f_mask = voxel_grid.check_if_included(o3d.utility.Vector3dVector(grid))
        valid_grid = grid[f_mask]
        area = len(valid_grid) * cell_a

        # Level synthetic floor: median ground-model elevation across the
        # footprint - a typical/representative reading of the diagnostic
        # ground surface under this building, not its lowest corner.
        floor_samples = ground_model.get_z(valid_grid[:, 0], valid_grid[:, 1])
        house_floor_z = np.median(floor_samples)

        # Bin f_pts into the exact same cells `grid`/`valid_grid` were built
        # from (searchsorted against xs/ys directly, not a reconstructed
        # mins+k*res, since np.arange's accumulated float values can drift
        # slightly from that - verified bin-for-bin identical against the
        # original loop across 200 randomized trials), then compute the
        # per-cell height with bincount/np.maximum.at instead of a Python loop
        # rebuilding a boolean mask over every point for every cell
        # (O(cells x points) -> O(points); ~76x faster measured on a
        # realistic single-house footprint).
        nx_bins, ny_bins = len(xs), len(ys)
        fx_idx = np.clip(np.searchsorted(xs, f_pts[:, 0], side='right') - 1, 0, nx_bins - 1)
        fy_idx = np.clip(np.searchsorted(ys, f_pts[:, 1], side='right') - 1, 0, ny_bins - 1)
        flat_idx = fx_idx * ny_bins + fy_idx

        # Per-cell MAX height, not mean: a cell straddling the roofline and a
        # wall below it should report the roof surface height, not an
        # average dragged down by wall points - mean systematically
        # under-estimates volume (worse in perimeter cells, and worse the
        # better the wall coverage gets, since more wall points pull the
        # mean down further).
        cell_max = np.full(nx_bins * ny_bins, -np.inf)
        np.maximum.at(cell_max, flat_idx, f_pts[:, 2])
        cell_max = cell_max.reshape(nx_bins, ny_bins)
        occupied = np.isfinite(cell_max)

        # Cells with no points at all (occlusion, low-texture feature-matching
        # gaps, or just sparse sampling) fall back to the NEAREST OCCUPIED
        # cell's height rather than one whole-building average - preserves
        # local roof shape (e.g. a short garage section vs. a tall main
        # section) instead of blending both into a single number. Same
        # distance_transform_edt technique LocalGroundModel already uses for
        # its own empty-cell fill.
        if not occupied.all():
            _, nearest_idx = distance_transform_edt(~occupied, return_distances=True, return_indices=True)
            cell_max = cell_max[tuple(nearest_idx)]

        grid_ix = np.clip(np.searchsorted(xs, valid_grid[:, 0], side='right') - 1, 0, nx_bins - 1)
        grid_iy = np.clip(np.searchsorted(ys, valid_grid[:, 1], side='right') - 1, 0, ny_bins - 1)
        vol = float(np.sum(cell_a * np.maximum(0, cell_max[grid_ix, grid_iy] - house_floor_z)))

        floor_pts = np.column_stack([valid_grid[:, 0], valid_grid[:, 1], np.full(len(valid_grid), house_floor_z)])
        floor_cols = np.full((len(floor_pts), 3), [0.4, 0.4, 0.4])
        combined_pts, combined_cols = np.vstack([f_pts, floor_pts]), np.vstack([f_cols, floor_cols])
    else:
        return None

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined_pts))
    pcd.colors = o3d.utility.Vector3dVector(combined_cols)
    # Floor points are always the trailing rows of combined_pts/pcd.points -
    # the caller uses this count to re-level them again after un-rotating
    # back to world coordinates (see worker_extraction).
    return pcd, area, vol, len(floor_pts)

def load_opensfm_camera(reconstruction, image_filename):
    """Takes an already-parsed reconstruction.json dict (reconstruction[0], not
    the file path) - this used to open and JSON-parse the file fresh on every
    call, which happened once per house; the file is 25+ MB on a real project,
    so that was 25+ MB of repeated parsing per house for data that never
    changes across the whole run."""
    if image_filename not in reconstruction['shots']:
        raise ValueError(f"Image '{image_filename}' not found in reconstruction data.")

    shot = reconstruction['shots'][image_filename]
    camera_id = shot['camera']
    camera = reconstruction['cameras'][camera_id]

    rvec = np.array(shot['rotation'], dtype=float)
    tvec = np.array(shot['translation'], dtype=float)

    w = camera['width']
    h = camera['height']
    max_size = max(w, h)

    focal = camera.get('focal', 0.8)
    fx = focal * max_size
    fy = focal * max_size

    c_x = camera.get('c_x', 0.0)
    c_y = camera.get('c_y', 0.0)
    cx = (w / 2.0) + (c_x * max_size)
    cy = (h / 2.0) + (c_y * max_size)

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=float)

    dist_coeffs = np.array([
        camera.get('k1', 0.0),
        camera.get('k2', 0.0),
        camera.get('p1', 0.0),
        camera.get('p2', 0.0),
        camera.get('k3', 0.0)
    ], dtype=float)

    return rvec, tvec, camera_matrix, dist_coeffs, w, h

# How many GPS-nearest candidate photos to try per house before giving up on a
# precision crop. Picking on GPS proximity alone (the old behavior) doesn't
# check that a photo was actually registered by OpenSfM, or that the house's
# footprint lands anywhere inside that photo's frame - either miss produced a
# real crop failure ("not found in reconstruction data" / an empty-slice
# imwrite assertion). Trying a handful of nearby candidates in distance order
# and taking the first one that clears both checks fixes both failure modes
# without having to guess in advance which single photo will work.
K_SOURCE_IMAGE_CANDIDATES = 8

def project_house_into_image(corners_3d, pose, padding=60):
    """Projects a house's 3D bounding box into one photo and returns the
    padded, frame-clamped crop box - or None if the house doesn't actually
    fall inside this photo (the two land entirely apart, e.g. the photo was
    the nearest by GPS position but pointed the wrong way, or the drone was
    just passing by). Returning None here, rather than clamping to a
    zero-or-negative-size box, is what lets the caller move on to the next
    candidate instead of handing an empty slice to cv2.imwrite."""
    rvec, tvec, K, dist, w, h = pose
    corners_2d, _ = cv2.projectPoints(np.array(corners_3d), rvec, tvec, K, dist)
    corners_2d = corners_2d.reshape(-1, 2)

    x_min, x_max = int(np.floor(np.min(corners_2d[:, 0]))), int(np.ceil(np.max(corners_2d[:, 0])))
    y_min, y_max = int(np.floor(np.min(corners_2d[:, 1]))), int(np.ceil(np.max(corners_2d[:, 1])))
    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max

def find_best_registered_image(gxy, corners_3d, img_data, reconstruction_data):
    """Tries the K nearest-by-GPS candidate photos in distance order and
    returns the first one that's both registered in the reconstruction and
    actually frames the house, along with its ready-to-use crop box. Returns
    (None, None) if none of the K candidates qualify - the caller falls back
    to whatever plain-nearest photo was already copied at extraction time
    (uncropped), so a run still ends with a source photo per house even when
    no candidate can be precision-cropped."""
    paths, tree = img_data
    if tree is None or reconstruction_data is None or corners_3d is None:
        return None, None

    k = min(K_SOURCE_IMAGE_CANDIDATES, len(paths))
    _, idxs = tree.query(gxy, k=k)
    idxs = np.atleast_1d(idxs)

    for i_idx in idxs:
        img_p = paths[i_idx]
        try:
            pose = load_opensfm_camera(reconstruction_data, os.path.basename(img_p))
        except ValueError:
            continue                              # not registered - try the next candidate
        box = project_house_into_image(corners_3d, pose)
        if box is not None:
            return img_p, box
    return None, None

# --- 3. PARALLEL WORKER ---
_worker_ctx = {}

def _init_worker(ground_model, rot, off_xy, g_off, img_data, out_dir):
    """ProcessPoolExecutor initializer: runs once per worker process at pool
    startup, not once per house. Data that's identical across every house in
    the run (ground_model's grid+interpolator, img_data's KDTree+paths, etc.)
    used to ride along in every single worker_args tuple, so it was pickled
    and sent through the task queue once per house; with N houses and only
    min(cores, 32) worker processes, this cuts that down to once per worker."""
    _worker_ctx['ground_model'] = ground_model
    _worker_ctx['rot'] = rot
    _worker_ctx['off_xy'] = off_xy
    _worker_ctx['g_off'] = g_off
    _worker_ctx['img_data'] = img_data
    _worker_ctx['out_dir'] = out_dir

def worker_extraction(args):
    idx, seeds, local_pts, local_cols = args
    ground_model = _worker_ctx['ground_model']
    rot = _worker_ctx['rot']
    off_xy = _worker_ctx['off_xy']
    g_off = _worker_ctx['g_off']
    img_data = _worker_ctx['img_data']
    out_dir = _worker_ctx['out_dir']

    # Standard 20cm noise filter above the true local ground elevation
    local_ground = ground_model.get_z(local_pts[:, 0], local_pts[:, 1])
    local_mask = local_pts[:, 2] > (local_ground + 0.20)

    local_pts, local_cols = local_pts[local_mask], local_cols[local_mask]
    if occupied_area_sqft(local_pts[:, :2]) < MIN_CANDIDATE_AREA_SQFT:
        return []

    valid_roof_pts = []
    rem = seeds.copy()
    while occupied_area_sqft(rem[:, :2]) > MIN_CANDIDATE_AREA_SQFT:
        tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rem))
        _, inliers = tmp.segment_plane(0.25, 3, 250)
        if occupied_area_sqft(rem[inliers][:, :2]) < MIN_CANDIDATE_AREA_SQFT:
            break
        valid_roof_pts.append(rem[inliers])
        rem = np.delete(rem, inliers, axis=0)

    if not valid_roof_pts:
        return []
    pure = np.vstack(valid_roof_pts)
    # min_samples is a raw count, deliberately left that way - unlike the
    # thresholds above, this one IS meant to be a local-density criterion
    # (that's what DBSCAN's min_samples parameter is for: distinguishing a
    # dense cluster from sparse noise). Converting it to an area-based check
    # would remove the density sensitivity it's supposed to have. In
    # practice it's rarely the binding constraint anyway - even at ~30
    # pts/m^2, a 2m-radius circle (eps) averages ~375 points, well above 15.
    b_labels = DBSCAN(eps=2.0, min_samples=15).fit(pure[:, :2]).labels_

    houses = []
    for b_id in range(b_labels.max() + 1):
        footprint = calculate_alpha_shape(pure[b_labels == b_id][:, :2])
        if not footprint:
            continue
        res = apply_house_cookie_cutter(local_pts, local_cols, footprint, ground_model)
        if res:
            p, a, v, n_floor = res
            # Undo the leveling transform in the same order it was applied in
            # reverse: shift XY back to world coordinates first, then un-rotate
            # about the origin. Rotating before translating would leak the XY
            # offset into Z, since the leveling rotation axis is horizontal.
            p.translate((off_xy[0], off_xy[1], 0))
            p.rotate(rot.T, center=(0, 0, 0))

            # The floor was level in the internal leveling frame, but that
            # frame's rotation is fit from the whole (possibly sloped) site,
            # not this one house - un-rotating it back to world coordinates
            # can re-introduce a tilt into what was a flat floor. Re-flatten
            # the floor to its own average elevation in world coordinates,
            # which is what "level" actually needs to mean here.
            if n_floor > 0:
                pts_arr = np.asarray(p.points)
                pts_arr[-n_floor:, 2] = pts_arr[-n_floor:, 2].mean()
                p.points = o3d.utility.Vector3dVector(pts_arr)

            z_vals = np.asarray(p.points)[:, 2]
            # 99th percentile instead of max: a single stray noise point
            # above the real roofline (a bird, a reflection artifact) would
            # otherwise inflate height directly - and height feeds the
            # area/height ratio used to keep/discard candidates in step 9, so
            # one outlier point could flip that decision. The floor is a
            # large, deliberate flat cluster (not a rare tail value), so the
            # low end doesn't have the same failure mode - left as min().
            height_m = np.percentile(z_vals, 99) - z_vals.min()
            height_ft = height_m * 3.28084
            area_sqft = a * 10.76
            vol_cuft = v * 35.31

            ratio = area_sqft / height_ft if height_ft > 0 else 0

            cent = np.mean(np.asarray(p.points), axis=0)
            gx, gy = cent[0] + g_off[0], cent[1] + g_off[1]

            temp_uid = f"H_{abs(gx):.3f}_{abs(gy):.3f}".replace('.', 'd')

            orig_img_name = "N/A"
            if img_data[1]:
                _, i_idx = img_data[1].query([gx, gy], k=1)
                img_p = img_data[0][i_idx]
                orig_img_name = os.path.basename(img_p)
                shutil.copy2(img_p, os.path.join(out_dir, "best_images", f"{temp_uid}.jpg"))

            o3d.io.write_point_cloud(os.path.join(out_dir, "individual_houses", f"{temp_uid}.ply"), p)

            # Compute the 3D bounding box here, while p is already in memory -
            # step 10 used to re-read this exact .ply back off disk per house
            # just to get this. p is in the same (world) coordinate frame the
            # saved .ply has, so this is exactly equivalent.
            corners_3d = np.asarray(p.get_axis_aligned_bounding_box().get_box_points())

            houses.append({
                "temp_ID": temp_uid,
                "Area_sqft": round(area_sqft, 2),
                "Volume_cuft": round(vol_cuft, 2),
                "Height_ft": round(height_ft, 2),
                "Ratio_Area_to_Height": round(ratio, 2),
                "X_coord": gx,
                "Y_coord": gy,
                "Original_Image": orig_img_name,
                "corners_3d": corners_3d.tolist()
            })
    return houses

# --- 4. MASTER PIPELINE ---
def process_reconstruction(args):
    global_start_time = time.time()
    project_path = os.path.abspath(args.project_path)
    project_name = os.path.basename(os.path.normpath(project_path))
    out = os.path.join(project_path, f"analysis_{project_name}_groundfloor")

    for d in ["individual_houses", "best_images", "best_images_cropped", "diagnostics"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    print(f"Output directory: {out}")

    print(f"CPU cores available: {NUM_CORES} (source: {_CORE_SOURCE})")
    mem_monitor = MemoryMonitor()
    if mem_monitor.start():
        print("Memory monitor: sampling via psutil (main + child worker processes)")
    else:
        print("Memory monitor: psutil not installed, falling back to a cruder end-of-run estimate "
              "(pip install psutil for an accurate number)")

    print("\n[1/10] Syncing Global Offsets...")
    g_off = auto_detect_offsets(project_path)
    reverse_trans = Transformer.from_crs(EPSG_CODE, "EPSG:4326", always_xy=True)
    img_data = build_spatial_image_index(os.path.join(project_path, 'images'), Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True))

    print("\n[2/10] Loading and cleaning PLY...")
    ply_path = os.path.join(project_path, 'scene_dense.ply')
    print(f"      -> Reading: {ply_path}")
    pcd = o3d.io.read_point_cloud(repair_openmvs_ply_colors(ply_path))
    print(f"      -> Raw point count: {len(pcd.points):,}")
    pcd = pcd.voxel_down_sample(0.05)
    print(f"      -> After 5cm voxel downsample: {len(pcd.points):,}")
    pcd, _ = pcd.remove_statistical_outlier(20, 2.2)
    print(f"      -> After statistical outlier removal: {len(pcd.points):,}")

    print("\n[3/10] Centering scene geometry...")
    pts = np.asarray(pcd.points)
    grnd = pts[pts[:, 2] < np.percentile(pts[:, 2], 30)]

    rot = np.eye(3)
    if args.relevel:
        # Opt-in only. auto_reconstruct.py's reconstruction_leveling.py
        # already applies a real gravity correction (DJI gimbal telemetry or
        # GPS-position alignment) upstream, before densification - fitting a
        # plane to the lowest 30% of points and forcing it flat here on top
        # of that is redundant at best. On a genuinely sloped site it's
        # actively wrong: that "lowest 30%" plane IS the real terrain slope,
        # and flattening it throws away the gravity reference already
        # established upstream. Only turn this on for point clouds that did
        # NOT go through the updated auto_reconstruct.py.
        _, evecs = np.linalg.eigh(np.cov(grnd.T))
        norm = evecs[:, 0] if evecs[2, 0] > 0 else -evecs[:, 0]
        v = np.cross(norm, [0, 0, 1])
        s, c = np.linalg.norm(v), np.dot(norm, [0, 0, 1])
        if s > 1e-6:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
        tilt_deg = np.degrees(np.arctan2(s, c))
        print(f"      -> --relevel active: rotating {tilt_deg:.2f} degrees to force the "
              f"lowest-30%-of-points plane horizontal")
    else:
        print("      -> Trusting upstream gravity correction (auto_reconstruct.py) - no additional "
              "rotation applied. Pass --relevel if this point cloud did NOT go through the updated "
              "auto_reconstruct.py.")

    pcd.rotate(rot, center=(0,0,0))
    off_xy = np.mean(grnd, axis=0)[:2]
    pcd.translate((-off_xy[0], -off_xy[1], 0))
    print(f"      -> Scene re-centered on XY offset: ({off_xy[0]:.3f}, {off_xy[1]:.3f})")

    print(f"\n[4/10] Building Local Contour Ground Model "
          f"(cell={args.ground_cell_size}m, opening={args.ground_opening_span}m)...")
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    ground_model = LocalGroundModel(pts, cell_size=args.ground_cell_size, opening_span=args.ground_opening_span)
    ground_model.write_diagnostic_ply(os.path.join(out, "diagnostics", "ground_surface.ply"), off_xy, rot)

    print("\n[5/10] Above-Ground Filter (AGL cut against local ground)...")
    ground_z_array = ground_model.get_z(pts[:, 0], pts[:, 1])
    # Strict AGL cut: keep points 0.35m (~1.1ft) above true local ground
    ag_mask = pts[:, 2] > (ground_z_array + 0.35)
    ag_pts = pts[ag_mask]
    ag_colors = cols[ag_mask]
    print(f"      -> {len(ag_pts):,} / {len(pts):,} points survive as above-ground candidates")

    ag_pcd = o3d.geometry.PointCloud()
    ag_pcd.points = o3d.utility.Vector3dVector(ag_pts)
    ag_pcd.colors = o3d.utility.Vector3dVector(ag_colors)
    ag_pcd, _ = ag_pcd.remove_statistical_outlier(12, 1.2)
    ag_pts, ag_colors = np.asarray(ag_pcd.points), np.asarray(ag_pcd.colors)
    print(f"      -> {len(ag_pts):,} points after statistical outlier removal")

    print("\n[6/10] Standard Geometric Consistency (vegetation removal)...")
    # Height-above-local-ground (not raw Z) so the statistical outlier test
    # still makes sense on sloped/uneven sites.
    ground_at_ag = ground_model.get_z(ag_pts[:, 0], ag_pts[:, 1])
    agl = ag_pts[:, 2] - ground_at_ag
    mean_h, std_h = np.mean(agl), np.std(agl)
    r, g, b = ag_colors[:, 0], ag_colors[:, 1], ag_colors[:, 2]
    exg = (2 * g) - r - b
    cand_idx = np.where((exg > 0.07) | ((agl - mean_h) / std_h > 2.5))[0]
    tree_mask = (exg > 0.07).copy()
    print(f"      -> {len(cand_idx):,} candidate vegetation/outlier points flagged for neighborhood check")

    if len(cand_idx) > 0:
        tree = KDTree(ag_pts)
        # Fixed-count neighbor query (not a fixed-radius one): a query_ball_point
        # expansion here scales with local point density as well as candidate
        # count, so it goes quadratic as density increases (e.g. from a higher
        # --depthmap-resolution reconstruction) - k=32 keeps per-candidate cost
        # bounded regardless of density.
        # Deliberately NOT converted to an area/density-based equivalent like the
        # count thresholds below - switching this back to a radius-based query
        # would reintroduce exactly the O(n^2) memory blowup that caused the
        # 360GB OOM crash this fixed-k query was originally written to solve.
        _, n_idx_l = tree.query(ag_pts[cand_idx], k=32, workers=-1)
        z_v = ag_pts[n_idx_l][:, :, 2]
        is_bumpy = (np.ptp(z_v, axis=1) > 0.35) | (np.std(z_v, axis=1) > 0.08)
        tree_mask[cand_idx[is_bumpy]] = True

    pruned_ag_pcd = ag_pcd.select_by_index(np.where(~tree_mask)[0])
    ag_pts, ag_colors = np.asarray(pruned_ag_pcd.points), np.asarray(pruned_ag_pcd.colors)
    print(f"      -> {int(tree_mask.sum()):,} points removed as vegetation/noise; {len(ag_pts):,} remain")

    print("\n[7/10] Morphological Erosion & Fragment Reconciliation...")
    final_pts = np.asarray(pruned_ag_pcd.points)

    # Query ground below surviving points to place seeds perfectly on roofs
    local_ground_z = ground_model.get_z(final_pts[:, 0], final_pts[:, 1])
    seeds = final_pts[final_pts[:, 2] > (local_ground_z + 2.2)]
    print(f"      -> {len(seeds):,} roof-seed points (>2.2m above local ground)")

    res = 0.25
    min_b, max_b = seeds[:, :2].min(axis=0), seeds[:, :2].max(axis=0)
    grid_shape = np.ceil((max_b - min_b) / res).astype(int) + 1
    coords = np.floor((seeds[:, :2] - min_b) / res).astype(int)

    occupancy_grid = np.zeros(grid_shape, dtype=bool)
    occupancy_grid[coords[:, 0], coords[:, 1]] = True

    eroded_grid = binary_erosion(occupancy_grid, iterations=3)
    labeled_grid, num_buildings = label(eroded_grid)
    print(f"      -> Erosion found {num_buildings} initial building cores")

    core_mask = eroded_grid[coords[:, 0], coords[:, 1]]
    core_pts = seeds[core_mask]
    core_labels = labeled_grid[coords[core_mask, 0], coords[core_mask, 1]]

    if occupied_area_sqft(core_pts[:, :2]) < MIN_CANDIDATE_AREA_SQFT:
        print("      [!] Error: Erosion removed all structures.")
        return

    core_tree = KDTree(core_pts[:, :2])
    dists, nearest_core_idx = core_tree.query(seeds[:, :2], k=1, workers=-1)

    labels = core_labels[nearest_core_idx]
    labels[dists > 2.0] = 0

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    label_sizes = {}
    label_coords = {}
    trees = {}

    for lbl in unique_labels:
        lbl_pts = seeds[labels == lbl]
        lbl_c = np.floor((lbl_pts[:, :2] - min_b) / res).astype(int)
        area_sqft = len(np.unique(lbl_c, axis=0)) * (res ** 2) * 10.764
        label_sizes[lbl] = area_sqft
        label_coords[lbl] = lbl_pts[:, :2]
        trees[lbl] = KDTree(label_coords[lbl])

    parent = {lbl: lbl for lbl in unique_labels}
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            if label_sizes[root_i] < label_sizes[root_j]:
                parent[root_i] = root_j
                label_sizes[root_j] += label_sizes[root_i]
            else:
                parent[root_j] = root_i
                label_sizes[root_i] += label_sizes[root_j]

    merge_count = 0
    for lbl_a in unique_labels:
        for lbl_b in unique_labels:
            if lbl_a >= lbl_b: continue
            root_a, root_b = find(lbl_a), find(lbl_b)
            if root_a == root_b: continue

            size_a, size_b = label_sizes[root_a], label_sizes[root_b]
            is_a_frag = size_a < 150 or size_a < (0.05 * size_b)
            is_b_frag = size_b < 150 or size_b < (0.05 * size_a)

            if not (is_a_frag or is_b_frag): continue

            dists_ab, _ = trees[lbl_b].query(label_coords[lbl_a], k=1, distance_upper_bound=0.75)
            if np.any(dists_ab < 0.75):
                union(lbl_a, lbl_b)
                merge_count += 1

    for i in range(len(labels)):
        if labels[i] > 0:
            labels[i] = find(labels[i])

    final_unique_labels = np.unique(labels)
    final_unique_labels = final_unique_labels[final_unique_labels > 0]
    print(f"      -> {merge_count} fragment(s) merged into larger neighbors; "
          f"{len(final_unique_labels)} candidates remain")

    worker_args = []
    for i in final_unique_labels:
        idx = np.where(labels == i)[0]
        blob = seeds[idx]
        # Reuses label_sizes[i] (already computed, already merged) rather than
        # a fresh point count - i is a root label here, so this is the same
        # area_sqft the fragment-reconciliation logic above already settled
        # on for this (possibly merged) blob. Matches that logic's own 150
        # sqft bar for consistency, rather than a second, different-meaning
        # "150" that used to be a raw point count.
        if label_sizes[i] < 150: continue
        m, M = blob.min(axis=0), blob.max(axis=0)
        mask = (ag_pts[:, 0] >= m[0]-5.0) & (ag_pts[:, 0] <= M[0]+5.0) & (ag_pts[:, 1] >= m[1]-5.0) & (ag_pts[:, 1] <= M[1]+5.0)
        worker_args.append((i, blob, ag_pts[mask], ag_colors[mask]))

    print(f"\n[8/10] Parallel Extraction ({len(worker_args)} candidates across {min(NUM_CORES, 32)} workers)...")
    phase_start = time.time()
    raw_res = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(NUM_CORES, 32),
        initializer=_init_worker,
        initargs=(ground_model, rot, off_xy, g_off, img_data, out)
    ) as ex:
        for res_list in ex.map(worker_extraction, worker_args):
            if res_list: raw_res.extend(res_list)
    print(f"      -> Extraction produced {len(raw_res)} raw building candidates in {time.time()-phase_start:.1f}s")

    print("\n[9/10] Artifact Purge & Sequential Georeferencing...")
    final_measurements = []
    lookup_table = []
    house_corners = {}  # house_ID -> 3D bounding box corners (computed once in the worker;
                         # step 10 uses this instead of re-reading each house's .ply from disk)
    house_counter = 1

    if len(raw_res) > 0:
        ratios = np.array([r["Ratio_Area_to_Height"] for r in raw_res])
        q1, q3 = np.percentile(ratios, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        eval_ratios = ratios[ratios <= upper_bound]

        if len(eval_ratios) >= 2:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(eval_ratios.reshape(-1, 1))
            natural_break = np.mean(kmeans.cluster_centers_.flatten())
            threshold = min(natural_break, 20.0)
        else:
            threshold = 10.0
        print(f"      -> Area/Height ratio threshold: {threshold:.2f} (upper_bound={upper_bound:.2f})")

        for r in raw_res:
            old_ply_path = os.path.join(out, "individual_houses", f"{r['temp_ID']}.ply")
            old_img_path = os.path.join(out, "best_images", f"{r['temp_ID']}.jpg")

            if r["Ratio_Area_to_Height"] > upper_bound or r["Ratio_Area_to_Height"] >= threshold:
                new_uid = f"{project_name}_{house_counter}"
                house_counter += 1

                new_ply_path = os.path.join(out, "individual_houses", f"{new_uid}.ply")
                new_img_path = os.path.join(out, "best_images", f"{new_uid}.jpg")

                if os.path.exists(old_ply_path): os.rename(old_ply_path, new_ply_path)
                if os.path.exists(old_img_path): os.rename(old_img_path, new_img_path)

                lon, lat = reverse_trans.transform(r['X_coord'], r['Y_coord'])

                final_measurements.append({
                    "house_ID": new_uid,
                    "Area_sqft": r["Area_sqft"],
                    "Volume_cuft": r["Volume_cuft"],
                    "Height_ft": r["Height_ft"],
                    "Ratio_Area_to_Height": r["Ratio_Area_to_Height"],
                    "Best_Image": new_img_path if os.path.exists(new_img_path) else "N/A"
                })

                lookup_table.append({
                    "house_ID": new_uid,
                    "X_UTM": round(r['X_coord'], 3),
                    "Y_UTM": round(r['Y_coord'], 3),
                    "Latitude": round(lat,6),
                    "Longitude": round(lon,6),
                    "Original_Image": r.get("Original_Image", "N/A")
                })
                house_corners[new_uid] = r.get("corners_3d")
            else:
                if os.path.exists(old_ply_path): os.remove(old_ply_path)
                if os.path.exists(old_img_path): os.remove(old_img_path)

    print(f"      -> Kept {len(final_measurements)} / {len(raw_res)} candidates as valid buildings")

    print("\n[10/10] Precision Cropping Images...")
    reconstruction_file = os.path.join(project_path, 'reconstruction.json')
    cropped_count = 0

    # Parse once - this used to happen fresh inside the loop below, once per
    # house. reconstruction.json only holds camera poses (doesn't change
    # per-house), and it's 25+ MB on a real project.
    reconstruction_data = None
    if os.path.exists(reconstruction_file):
        with open(reconstruction_file, 'r') as f:
            reconstruction_data = json.load(f)[0]

    no_candidate_count = 0
    for record in lookup_table:
        house_id = record["house_ID"]
        if record["Original_Image"] == "N/A": continue

        crop_out_path = os.path.join(out, "best_images_cropped", f"{house_id}.jpg")
        corners_3d = house_corners.get(house_id)
        gxy = [record["X_UTM"], record["Y_UTM"]]

        chosen_path, box = find_best_registered_image(gxy, corners_3d, img_data, reconstruction_data)

        if chosen_path is None:
            # None of the K nearest candidates were both registered and
            # actually framed the house - leave whatever plain-nearest photo
            # extraction already copied to best_images/{house_id}.jpg in
            # place (uncropped) rather than fail with nothing at all.
            no_candidate_count += 1
            continue

        try:
            img = cv2.imread(chosen_path)
            if img is None:
                raise ValueError(f"cv2 could not read {chosen_path}")
            x_min, y_min, x_max, y_max = box

            # Refresh best_images/ to the SAME photo the crop came from - the
            # candidate that wins here can differ from extraction time's
            # naive plain-nearest pick, and the two should always agree.
            shutil.copy2(chosen_path, os.path.join(out, "best_images", f"{house_id}.jpg"))
            cv2.imwrite(crop_out_path, img[y_min:y_max, x_min:x_max])
            record["Original_Image"] = os.path.basename(chosen_path)
            cropped_count += 1
        except Exception as e:
            print(f"      [!] Failed to crop {house_id}: {e}")

    print(f"      -> Cropped {cropped_count} / {len(lookup_table)} source images"
          f" ({no_candidate_count} had no registered/in-frame candidate among "
          f"the {K_SOURCE_IMAGE_CANDIDATES} nearest - left uncropped)")

    measurements_path = os.path.join(out, "measurements.csv")
    with open(measurements_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["house_ID", "Area_sqft", "Volume_cuft", "Height_ft", "Ratio_Area_to_Height", "Best_Image"])
        writer.writeheader()
        writer.writerows(final_measurements)
    print(f"      -> Measurements written to: {measurements_path}")
    print(f"      -> NOTE: floor is the MEDIAN ground-model elevation across each footprint "
          f"(see diagnostics/ground_surface.ply), not the minimum.")

    lookup_path = os.path.join(out, "location_lookup.csv")
    with open(lookup_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["house_ID", "X_UTM", "Y_UTM", "Latitude", "Longitude", "Original_Image"])
        writer.writeheader()
        writer.writerows(lookup_table)
    print(f"      -> Location lookup written to: {lookup_path}")

    peak_gb, peak_method = mem_monitor.stop_and_report()

    total_elapsed = time.time() - global_start_time
    print("-" * 60)
    print(f"SUCCESS. Validated and cropped {len(final_measurements)} buildings in {total_elapsed:.2f}s")
    print(f"Results directory: {out}")
    print(f"Peak memory used: {peak_gb:.1f} GB  ({peak_method})")
    print(f"CPU cores used: {NUM_CORES} (source: {_CORE_SOURCE})")
    print(f"Point cloud size this run: {len(pts):,} points (from scene_dense.ply)")
    print("Keep a note of the two lines above against this dataset's size - they're the real")
    print("numbers to use for --mem / --cpus-per-task next time you run something similar,")
    print("more reliable than any pre-flight estimate.")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract buildings from point cloud data - floor is the median ground-model elevation per footprint.")
    parser.add_argument("project_path", type=str, help="Path to the OpenDroneMap project directory.")
    parser.add_argument("--ground-cell-size", type=float, default=2.0,
                        help="Grid cell size in meters for the local contour ground model. Smaller = more "
                             "detail but noisier; larger = smoother but coarser. Default 2.0m.")
    parser.add_argument("--ground-opening-span", type=float, default=20.0,
                        help="Morphological opening span in meters. Must be noticeably larger than the "
                             "largest building footprint in the scene, or roofs will still be misread as "
                             "ground. Default 20.0m.")
    parser.add_argument("--relevel", action="store_true",
                        help="Additionally re-level via a PCA fit to the lowest 30%% of points, forcing that "
                             "plane horizontal. Off by default - auto_reconstruct.py's own gravity correction "
                             "(DJI/GPS-based) should already be trustworthy, and this local PCA fit would "
                             "double-correct on top of it, which is actively wrong on a genuinely sloped site "
                             "(the lowest-30%% plane there IS the real terrain slope). Only use this for point "
                             "clouds that did NOT go through the updated auto_reconstruct.py.")

    args = parser.parse_args()
    process_reconstruction(args)
