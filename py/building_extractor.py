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
from scipy.spatial import Delaunay, KDTree, ConvexHull
from scipy.ndimage import binary_erosion, label, grey_opening, grey_closing, gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pyproj import Transformer
import concurrent.futures
from resource_monitor import detect_cpu_count, MemoryMonitor
from reconstruction_leveling import read_dji_gimbal_attitude
from pipeline_config import load_config

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
        # terrain slope/hills untouched since they're much wider than a
        # building - EXCEPT this silently fails for any building WIDER than
        # opening_span: confirmed as a real bug on a 47x76m commercial
        # building, where every interior cell is >20m (opening_span's
        # default) from any real ground, so the erosion window can only
        # ever see OTHER roof cells - the model then believes the roof
        # height itself IS the ground there, and the above-ground filter
        # (correctly, given that wrong premise) strips the roof out of the
        # extraction entirely, leaving only a border ring where the
        # building happens to be within reach of real ground (confirmed
        # again on a real ~8500 sqft commercial roof on the Blockall site).
        #
        # Two different attempts at fixing this INSIDE the ground model were
        # tried and reverted, both for the same underlying reason: any
        # opening wide enough to reach past an oversized roof to real ground
        # is also wide enough to reach into unrelated terrain the grid has
        # no way to distinguish from "this building's own ground":
        #   1. Blind minimum-across-scales, applied everywhere - collapsed
        #      the ground model toward street level across ~97% of the whole
        #      grid on a real site, not just the one oversized building.
        #   2. A bounded, connected-component-gated version of the same idea
        #      (only rescue cells inside a patch under a hard span cap) -
        #      fixed the intended building, but in tightly-packed blocks the
        #      wide-reach RESCUE VALUE itself (not just which cells got
        #      rescued) could still reach past one building into a
        #      neighbor's yard, pulling the ground estimate down enough
        #      there that real gap/yard material started passing the above-
        #      ground filter and bridging separate buildings into one
        #      candidate - confirmed on a real dense block.
        #
        # Both failures came from doing this at the GRID level, where a
        # cell has no idea which specific building (if any) it belongs to.
        # The actual fix now lives at the CANDIDATE level instead -
        # rescue_ring_shaped_candidates(), called after extraction - which
        # only ever pulls in points that (a) fall within one already-
        # extracted candidate's own convex hull and (b) match THAT
        # candidate's own roof height, so it can't reach into a neighbor's
        # territory no matter how tightly packed the site is. This class
        # goes back to being exactly the original single-scale version
        # (byte-for-byte, confirmed against this project's own git history)
        # - the oversized-roof case is a known, real gap here, left for the
        # candidate-level pass to close instead of trying (again) to solve
        # it with a wider opening.
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
        print(f"      -> Opening kernel: {k}x{k} cells (~{opening_span:.0f}m span)")
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

# DJI convention: GimbalPitchDegree -90 = straight down, 0 = horizontal.
# The actual cutoff (imagery.nadir_pitch_threshold) lives in config.yml, not
# here - confirmed it's genuinely per-site: Blockall's two survey passes
# split cleanly at -75 (a steep -90 to -70 pass, 293 images, vs. a
# distinctly shallower pass clustered around -60, 127 images), but the fir
# dataset's steepest pass only reaches -72 and needs a different threshold
# to split into two groups at all. "Oblique" means "the shallower of a
# site's two passes" specifically, not a guaranteed side-on elevation shot -
# how dramatic that view is depends entirely on how shallow that site's own
# second pass was flown.

def build_dual_spatial_image_index(image_dir, transformer, nadir_pitch_threshold):
    """Like a single build_spatial_image_index used to work, but splits
    images into two separate GPS-KDTrees by DJI gimbal pitch: nadir (looking
    straight down) and oblique (the shallower pass, see nadir_pitch_threshold
    - config.yml's imagery.nadir_pitch_threshold, NOT a universal constant;
    confirmed the right cutoff varies by flight - Blockall's two passes
    split cleanly at -75, but the fir dataset's steepest pass only reaches
    -72 and needs a different threshold to split at all). An image with no
    readable gimbal telemetry (non-DJI drone, or tags missing) is classified
    as nadir - matches this project's existing reconstruction.depthmap_nadir_only
    behavior of treating unclassifiable shots as the safe/inclusive default
    rather than silently losing them.

    Returns (nadir_img_data, oblique_img_data), each a (paths, KDTree) pair
    or (None, None) if that category has no usable images - same shape
    build_spatial_image_index used to return, so find_best_registered_image
    works unchanged against either one."""
    paths_n, coords_n, paths_o, coords_o = [], [], [], []
    if not image_dir or not os.path.exists(image_dir):
        print(f"      [!] Image directory not found: {image_dir} (no source photos will be attached)")
        return (None, None), (None, None)
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
        except Exception:
            continue

        attitude = read_dji_gimbal_attitude(p)
        coord = transformer.transform(lon, lat)
        if attitude is not None and attitude[1] > nadir_pitch_threshold:
            paths_o.append(p)
            coords_o.append(coord)
        else:
            paths_n.append(p)
            coords_n.append(coord)

    print(f"      -> Indexed {len(paths_n)} nadir + {len(paths_o)} oblique image(s) with usable GPS EXIF "
          f"from {image_dir} (pitch threshold {nadir_pitch_threshold:.0f} deg, {len(files)} files total)")
    nadir_data = (paths_n, KDTree(np.array(coords_n))) if coords_n else (None, None)
    oblique_data = (paths_o, KDTree(np.array(coords_o))) if coords_o else (None, None)
    return nadir_data, oblique_data

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
# size depending on point density, which silently shifts with e.g. a higher
# reconstruction.depthmap_resolution - the same "100 points" that meant ~36 sqft (a shed)
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

        # A foreign low object sitting inside the footprint (a parked car
        # under an eave, a bush the alpha-shape happened to bulge over) is
        # OCCUPIED, not empty - the fill above never touches it, and its own
        # much-lower height goes straight into the volume/point-cloud output
        # as if it were roof. Detect this as a small, deep LOCAL dip:
        # grey_closing (dilate then erode) raises anything narrower than
        # `closing_cells` up to its surroundings' local max, so subtracting
        # the original from the closed version isolates exactly the cells
        # that got pulled up - i.e. cells surrounded by much taller neighbors
        # on most sides within that window. `closing_cells` is sized for a
        # car (~5.4m); a real lower roof section (a garage wing, a one-story
        # addition) is normally wider than that in both directions, so most
        # of it survives untouched - only its border toward the taller
        # section blurs slightly, an accepted, minor tradeoff of using a
        # fixed-size morphological filter for this rather than a full
        # connected-component analysis of what belongs to "the roof."
        structure_m, dip_margin_m = 5.4, 1.0
        closing_cells = max(3, int(round(structure_m / res)) | 1)  # odd size, so it's centered
        closed = grey_closing(cell_max, size=closing_cells)
        suspect = occupied & ((closed - cell_max) > dip_margin_m)
        if suspect.any():
            trustworthy = occupied & ~suspect
            if trustworthy.any():
                _, nearest_idx = distance_transform_edt(~trustworthy, return_distances=True, return_indices=True)
                cell_max = np.where(suspect, cell_max[tuple(nearest_idx)], cell_max)

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

# --- Vegetation-contamination review flag ---
# Always computed (no CLI gate - always on, for every house, every run).
#
# check_vegetation_contamination() below is a WEAKER signal than originally
# believed - keep this in mind before trusting it alone. It was built on the
# assumption that contamination generally looks color-different from a
# house's own roof material. Verified against two real, confirmed
# contamination cases and found to be unreliable in both directions: on one
# house, a large tree turned out to have ~23% of its points color-matching
# the real roof closely enough to fool a brightness-based read of the same
# scene (both directions - sunlit tree read as roof, shadowed roof read as
# tree); on another, the actual contaminating cluster's color (160,135,122)
# was nearly identical to the house's own roof average (160,129,128) - nothing
# about it looked anomalous by color at all. So this check is kept as a
# cheap, harmless SECOND opinion, not the primary signal - see
# filter_disconnected_fragments() below and its call sites for the stronger,
# validated one (how much of a candidate a 3D-connectivity pass had to
# remove), which the final Needs_Review decision is actually anchored on.
FLAG_SEED_PLANES = 2
FLAG_Z_THRESHOLD = 2.5
FLAG_MIN_FRACTION = 0.15
FLAG_MIN_SEED_POINTS = 200

def check_vegetation_contamination(pts, cols):
    """Returns True if a large fraction of `pts`/`cols` (a house's own
    above-ground points, WITHOUT its synthetic floor) doesn't color-match a
    RANSAC-seeded sample of its own dominant roof material. Self-calibrated
    per house rather than a fixed rule like the ExG color test in step 6,
    but see the module-level comment above this function before trusting
    it - color turned out to be an unreliable way to identify
    contamination in general; this is a supplementary check, not the
    primary one."""
    if len(pts) < FLAG_MIN_SEED_POINTS:
        return False

    rem = pts.copy()
    planes = []
    for _ in range(FLAG_SEED_PLANES):
        if occupied_area_sqft(rem[:, :2]) <= MIN_CANDIDATE_AREA_SQFT:
            break
        tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rem))
        plane_model, inliers = tmp.segment_plane(0.25, 3, 250)
        inlier_mask = np.zeros(len(rem), dtype=bool)
        inlier_mask[inliers] = True
        if occupied_area_sqft(rem[inlier_mask][:, :2]) < MIN_CANDIDATE_AREA_SQFT:
            break
        a, b, c, d = plane_model
        planes.append((np.array([a, b, c]), d))
        rem = rem[~inlier_mask]
    if not planes:
        return False

    dists = np.stack([np.abs(pts @ n + d) for n, d in planes], axis=1)
    seed_mask = dists.min(axis=1) < 0.3
    if seed_mask.sum() < FLAG_MIN_SEED_POINTS:
        return False

    seed_cols = cols[seed_mask].astype(float)
    mean_c, std_c = seed_cols.mean(axis=0), seed_cols.std(axis=0) + 1e-6
    z = np.sqrt((((cols.astype(float) - mean_c) / std_c) ** 2).sum(axis=1))
    return bool(np.mean(z > FLAG_Z_THRESHOLD) > FLAG_MIN_FRACTION)

# --- Possible-vehicle review flag ---
# A real rejection filter was attempted first, not this. Three independent
# geometric signals were tested against real confirmed vehicles and real
# confirmed buildings pulled from an actual run (Blockall dataset):
#   - RANSAC top-2-plane inlier fraction (hypothesis: a car's curved
#     hood/roof/windshield shouldn't fit two flat planes as well as a real
#     roof does): confirmed cars scored 0.586-0.847, confirmed houses
#     0.391-0.930 - total overlap, and one real house scored BELOW every
#     single car.
#   - Area_sqft and Height_ft, alone and combined: a confirmed car cluster
#     (522.9 sqft, 12.05 ft) and an unrelated, presumably legitimate small
#     candidate (557.8 sqft, 12.14 ft) came out nearly identical.
# None of them separate the two populations. This makes sense once you
# picture what a nadir point cloud actually captures: from directly
# overhead, both a small shed/garage and a parked truck reduce to the same
# thing - a compact, flat-topped blob of some height and area - and those
# size ranges genuinely overlap in the real world, not just in this
# dataset. Pure point-cloud geometry isn't the right tool for this; the
# actual distinguishing signal is visual (a car looks nothing like a roof
# in the cropped photo), which would need an image classifier, not another
# threshold here.
#
# So this is deliberately a REVIEW flag, not an auto-reject: a size/height
# ceiling comfortably above every confirmed car seen so far, meant to
# narrow down what to manually check rather than claim to know the answer.
# It WILL also flag some legitimate small structures (sheds, small
# garages) - that's expected and fine for a flag, not acceptable for a
# silent deletion. Thresholds live in config.yml (vehicle_rejection.
# max_area_sqft / max_height_ft) - calibrated against one real dataset, not
# a validated universal cutoff, so worth checking per site.
def check_possible_vehicle(area_sqft, height_ft, max_area_sqft, max_height_ft):
    return bool(area_sqft < max_area_sqft and height_ft < max_height_ft)

# --- Disconnected-fragment filter ---
# Color and per-cell height/thickness were both tried and confirmed unsafe
# as general contamination filters (see check_vegetation_contamination's
# comment above, and grey_closing dip-filter's opposite-direction limits).
# What's validated: 3D spatial connectivity. A tree pressed against a roof
# doesn't have to be one big blob to matter - a real confirmed case turned
# out to be ~40 separate small fragments that only looked like one cluster
# when viewed by height alone, not by 3D adjacency. Real 3D connected-
# components generalizes that "find the disconnected stuff" idea properly:
# it finds every separate piece, of any size, anywhere in the point cloud,
# rather than only the single biggest gap along one axis.
#
# The keep/discard rule is deliberately RELATIVE, not absolute: keep a
# component if its own occupied area is at least MIN_COMPONENT_FRACTION of
# the LARGEST component's area in this same candidate, discard the rest.
# This was necessary, not just simpler - confirmed on two real candidates
# that a component's absolute size can't tell "real secondary roof facet"
# apart from "small foreign structure": one candidate's legitimate second
# roof section (a real hip/wing, not touching the main roof closely enough
# to be one component) was 43.4% of its largest component's area; another
# candidate's contamination - a NEIGHBOR'S SHED, bridged into this
# candidate's footprint by a connecting tree during DBSCAN clustering, not
# just noise - was 4.8% of its largest. A threshold anywhere from 5% to 30%
# separates those two correctly; MIN_COMPONENT_FRACTION=0.10 sits in the
# middle of that margin.
#
# What this does NOT solve: a tree that's genuinely 3D-connected to the
# real roof (touching/overlapping in the reconstruction, not just nearby)
# survives as part of the same largest component - confirmed on the same
# real "massive tree" case, where this stage alone recovers the roof from
# 100% contaminated down to 87%, not further. That residual case has no
# known safe automatic fix yet (see the Needs_Review wiring at each call
# site) - it needs a human to look at it.
CONNECT_RADIUS = 0.15
MIN_COMPONENT_FRACTION = 0.10

# How much of a candidate filter_disconnected_fragments has to remove before
# that removal itself counts as review-worthy evidence, independent of the
# color check. Calibrated against the same two real cases as the fraction
# above: a fully-resolved candidate needed only ~3.3% removed (comfortably
# below), a candidate with a KNOWN unresolved remainder (a tree genuinely
# touching the roof, not just nearby) needed ~13% (comfortably above) -
# 0.05 sits in the middle of that margin.
STAGE1_REVIEW_FRACTION = 0.05

def filter_disconnected_fragments(pts, radius=CONNECT_RADIUS, min_fraction=MIN_COMPONENT_FRACTION):
    """3D connected-components at `radius`; keeps only components whose own
    occupied area is at least `min_fraction` of the largest component's
    area. Returns a boolean keep-mask the same length as `pts`."""
    n = len(pts)
    if n < 10:
        return np.ones(n, dtype=bool)
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')
    if len(pairs) == 0:
        return np.ones(n, dtype=bool)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp <= 1:
        return np.ones(n, dtype=bool)
    areas = np.array([occupied_area_sqft(pts[labels == lbl][:, :2]) for lbl in range(n_comp)])
    max_area = areas.max()
    if max_area <= 0:
        return np.ones(n, dtype=bool)
    keep_labels = np.where(areas >= min_fraction * max_area)[0]
    return np.isin(labels, keep_labels)

def finalize_candidate(ag_pts, ag_cols, footprint, ground_model, rot, off_xy):
    """Shared post-cookie-cutter pipeline for ONE candidate - used by both
    worker_extraction and merge_overlapping_footprints, which used to each
    carry their own copy of this logic. Centralizing it fixed two confirmed
    bugs that came from the two copies drifting:

    1. A stale, wider floor surviving underneath material the disconnected-
       fragment filter removed. apply_house_cookie_cutter's floor is sized
       for the FOOTPRINT it's given; if filter_disconnected_fragments then
       removes some roof/wall points, the OLD floor (still spanning the
       original, wider footprint) doesn't shrink to match - confirmed as a
       real bug: a large, isolated floor patch with no roof above it, in a
       candidate that had otherwise correctly merged. Fixed by re-tracing a
       fresh alpha-shape footprint from the SURVIVORS and re-running
       cookie-cutter whenever anything gets removed, rather than only
       patching the area/volume scalars.

    2. merge_overlapping_footprints used to run cookie-cutter (and its
       ground_model.get_z() floor-height lookups) directly on WORLD-frame
       points reloaded from disk, while ground_model is built in the
       rotated+translated INTERNAL frame - a real coordinate mismatch that
       could put a merged candidate's floor at the wrong elevation. Fixed by
       requiring ag_pts/footprint here to already be in ground_model's own
       internal frame (callers convert first if needed), with the
       world-frame conversion happening in exactly one place, at the end.

    Returns (p, area_m2, vol_m3, n_floor, needs_review_stage1) - p already
    converted to world coordinates and floor-releveled, ready for height/
    area/Needs_Review computation and writing to disk - or None if
    apply_house_cookie_cutter can't build a candidate from ag_pts/footprint
    at all."""
    res = apply_house_cookie_cutter(ag_pts, ag_cols, footprint, ground_model)
    if not res:
        return None
    p, a, v, n_floor = res

    all_pts, all_cols = np.asarray(p.points), np.asarray(p.colors)
    frag_pts = all_pts[:-n_floor] if n_floor > 0 else all_pts
    frag_cols = all_cols[:-n_floor] if n_floor > 0 else all_cols

    removed_frac = 0.0
    if n_floor > 0 and len(frag_pts) > 0:
        keep = filter_disconnected_fragments(frag_pts)
        frac = 1.0 - keep.mean()
        if frac > 0:
            filtered_pts, filtered_cols = frag_pts[keep], frag_cols[keep]
            new_footprint = calculate_alpha_shape(filtered_pts[:, :2])
            res2 = (apply_house_cookie_cutter(filtered_pts, filtered_cols, new_footprint, ground_model)
                    if new_footprint else None)
            if res2:
                p, a, v, n_floor = res2
                removed_frac = frac
            # else: re-tracing failed (too few points survived, or a
            # degenerate shape) - keep the PRE-filter p/a/v/n_floor rather
            # than ship a mismatched or missing floor for this rare case.

    # Undo the leveling transform in the same order it was applied in
    # reverse: shift XY back to world coordinates first, then un-rotate
    # about the origin. Rotating before translating would leak the XY
    # offset into Z, since the leveling rotation axis is horizontal.
    p.translate((off_xy[0], off_xy[1], 0))
    p.rotate(rot.T, center=(0, 0, 0))

    # The floor was level in the internal leveling frame, but that frame's
    # rotation is fit from the whole (possibly sloped) site, not this one
    # candidate - un-rotating it back to world coordinates can re-introduce
    # a tilt into what was a flat floor. Re-flatten it to its own average
    # elevation in world coordinates, which is what "level" actually needs
    # to mean here.
    if n_floor > 0:
        all_pts = np.asarray(p.points)
        all_pts[-n_floor:, 2] = all_pts[-n_floor:, 2].mean()
        p.points = o3d.utility.Vector3dVector(all_pts)

    needs_review_stage1 = removed_frac > STAGE1_REVIEW_FRACTION
    return p, a, v, n_floor, needs_review_stage1

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

def project_house_into_image(corners_3d, pose, padding=60):
    """Projects a house's 3D bounding box into one photo. Returns
    (crop_box, visible_area): crop_box is the padded, frame-clamped
    (x_min, y_min, x_max, y_max) ready to hand to cv2, and visible_area is
    the pixel area of the UNPADDED projected box that actually lands inside
    the frame - used by find_best_registered_image to rank candidate photos
    by how prominently they frame the house, not just whether they frame it
    at all. Returns (None, 0.0) if the house's footprint doesn't land inside
    this photo at all (e.g. the photo was GPS-near but pointed elsewhere),
    or if any corner of the box is behind the camera (negative depth) - a
    pinhole projection doesn't fail cleanly for points behind the lens, it
    produces wild garbage pixel coordinates (confirmed directly: one real
    case projected a corner to ~5e20), and those can spuriously clip into
    frame bounds and register as a huge, "prominent" match. A real,
    in-front-of-camera photo of the whole building can't have any of its own
    corners behind the lens, so this rejects the candidate outright rather
    than trust a projection that's partly not physically meaningful.

    Also rejects corners that are technically in front of the camera but
    far outside its actual field of view. Positive depth alone doesn't mean
    "in the shot" - a point 60+ degrees off the optical axis still has
    positive depth. cv2's polynomial distortion model isn't calibrated that
    far out and can fold such a point back into frame-bounded pixel
    coordinates that look like a real (if narrow) match. Confirmed directly
    on the fir dataset: a house 60+ degrees off-axis against this camera's
    own ~32 degree half-FOV still produced a "successful" crop - a 72px-wide
    sliver of unrelated content. The bound below is derived from each
    camera's own intrinsics (how far its own frame edge sits in normalized
    coordinates), not a hardcoded angle, so it scales correctly across
    different cameras/resolutions; the 1.5x margin allows real subjects
    legitimately near the frame edge without accepting 60+ degree outliers."""
    rvec, tvec, K, dist, w, h = pose
    corners_3d = np.asarray(corners_3d)
    rot_mat, _ = cv2.Rodrigues(rvec)
    cam_pts = rot_mat @ corners_3d.T + tvec.reshape(3, 1)
    depth = cam_pts[2]
    if np.any(depth <= 0):
        return None, 0.0

    norm_x, norm_y = cam_pts[0] / depth, cam_pts[1] / depth
    fov_x = 1.5 * max(K[0, 2], w - K[0, 2]) / K[0, 0]
    fov_y = 1.5 * max(K[1, 2], h - K[1, 2]) / K[1, 1]
    if np.any(np.abs(norm_x) > fov_x) or np.any(np.abs(norm_y) > fov_y):
        return None, 0.0

    corners_2d, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, dist)
    corners_2d = corners_2d.reshape(-1, 2)

    x_min, x_max = np.min(corners_2d[:, 0]), np.max(corners_2d[:, 0])
    y_min, y_max = np.min(corners_2d[:, 1]), np.max(corners_2d[:, 1])

    visible_w = max(0.0, min(float(w), x_max) - max(0.0, x_min))
    visible_h = max(0.0, min(float(h), y_max) - max(0.0, y_min))
    visible_area = visible_w * visible_h
    if visible_area <= 0.0:
        return None, 0.0

    px_min, py_min = max(0, int(np.floor(x_min)) - padding), max(0, int(np.floor(y_min)) - padding)
    px_max, py_max = min(w, int(np.ceil(x_max)) + padding), min(h, int(np.ceil(y_max)) + padding)
    if px_max <= px_min or py_max <= py_min:
        return None, 0.0
    return (px_min, py_min, px_max, py_max), visible_area

def find_best_registered_image(corners_3d, img_data, reconstruction_data):
    """Scores every registered candidate photo in img_data by how
    prominently it frames the house (visible projected area - see
    project_house_into_image) and returns the best-framed one, along with
    its ready-to-use crop box. Returns (None, None) if nothing in the pool
    frames the house at all - the caller falls back to whatever plain-
    nearest photo was already copied at extraction time (uncropped), so a
    run still ends with a source photo per house even then.

    Deliberately NOT "closest by GPS position, first candidate that
    technically overlaps" (the old behavior) - confirmed as a real bug on
    the fir dataset, shot obliquely across the street: camera GPS position
    says nothing about where the camera was pointed, so a photo taken right
    next to a DIFFERENT house can still technically graze this house's
    corners at the frame edge, win purely for being GPS-close, and get
    returned as this house's "best" photo even though a different, far
    better-framed photo of it exists elsewhere in the pool (real case: a
    frame 16m from house 1050 was returned as two other houses' "best oblique"
    despite those houses sitting 70+m away and having their own well-framed
    photos available). Scoring every registered candidate rather than just
    the GPS-nearest few is what finds those - the actual best oblique photo
    of a house shot across the street isn't necessarily GPS-near it at all.
    Cheap enough to do exhaustively: projecting 8 corners into a few hundred
    candidate photos per house is well under a second in aggregate."""
    paths, tree = img_data
    if tree is None or reconstruction_data is None or corners_3d is None:
        return None, None

    best_path, best_box, best_area = None, None, 0.0
    for img_p in paths:
        try:
            pose = load_opensfm_camera(reconstruction_data, os.path.basename(img_p))
        except ValueError:
            continue                              # not registered - try the next candidate
        box, area = project_house_into_image(corners_3d, pose)
        if box is not None and area > best_area:
            best_path, best_box, best_area = img_p, box, area
    return best_path, best_box

def try_precision_crop(house_id, view_name, corners_3d, img_data, reconstruction_data, out_dir):
    """Attempts a precision crop for one view ('nadir' or 'oblique') of one
    house against one image index. On success, writes
    best_images/{house_id}_{view_name}.jpg (refreshed to the winning photo)
    and best_images_cropped/{house_id}_{view_name}.jpg, and returns
    (source_photo_basename, True). Returns (None, False) if no candidate in
    img_data both registered and framed the house, or on any read/write
    error - the caller decides what that means for its own bookkeeping."""
    chosen_path, box = find_best_registered_image(corners_3d, img_data, reconstruction_data)
    if chosen_path is None:
        return None, False
    try:
        img = cv2.imread(chosen_path)
        if img is None:
            raise ValueError(f"cv2 could not read {chosen_path}")
        x_min, y_min, x_max, y_max = box
        shutil.copy2(chosen_path, os.path.join(out_dir, "best_images", f"{house_id}_{view_name}.jpg"))
        cv2.imwrite(os.path.join(out_dir, "best_images_cropped", f"{house_id}_{view_name}.jpg"),
                    img[y_min:y_max, x_min:x_max])
        return os.path.basename(chosen_path), True
    except Exception as e:
        print(f"      [!] Failed to crop {house_id} ({view_name}): {e}")
        return None, False

# --- 3. PARALLEL WORKER ---
# Caps how many points Open3D's RANSAC plane search itself has to churn
# through per call. worker_extraction below peels one dominant plane at a
# time off a candidate's seed points (see its own comment) - fine for an
# ordinary single roof (a handful of rounds), but a real run confirmed this
# loop taking 636s for just 18 candidates on a site with known giant multi-
# building/gutted-commercial blobs (35s/candidate vs 1.6-2.4s/candidate on
# cleaner sites) - those blobs hand segment_plane hundreds of thousands of
# points per round, and Open3D's own inlier-counting cost scales with
# however many points it's given, for EVERY one of its 250 RANSAC
# iterations. A plane's fitted parameters don't get more accurate past a
# few tens of thousands of representative points (RANSAC just needs enough
# samples to score candidate hypotheses) - the search cost does, linearly.
RANSAC_MAX_FIT_POINTS = 50000

def segment_plane_capped(pts, distance_threshold=0.25, ransac_n=3, num_iterations=250,
                          max_fit_points=RANSAC_MAX_FIT_POINTS):
    """Same (plane_model, inlier_indices_into_pts) contract as Open3D's
    PointCloud.segment_plane. Below max_fit_points, this is a byte-for-byte
    passthrough to Open3D's own call (zero behavior change for the common
    case - most candidates never hit this path at all). Above it, RANSAC's
    own search runs against a uniform random subsample capped at
    max_fit_points, and the returned plane model is then applied to the
    FULL `pts` with one cheap vectorized distance check to get the real
    inlier set - so which points end up counted as part of the plane
    doesn't change, only how expensively the model gets found."""
    if len(pts) <= max_fit_points:
        tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        return tmp.segment_plane(distance_threshold, ransac_n, num_iterations)

    idx = np.random.choice(len(pts), max_fit_points, replace=False)
    tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[idx]))
    plane_model, _ = tmp.segment_plane(distance_threshold, ransac_n, num_iterations)
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    dists = np.abs(pts @ normal + d) / np.linalg.norm(normal)
    inliers = np.where(dists < distance_threshold)[0]
    return plane_model, inliers

_worker_ctx = {}

def _init_worker(ground_model, rot, off_xy, g_off, nadir_img_data, out_dir, ransac_max_fit_points):
    """ProcessPoolExecutor initializer: runs once per worker process at pool
    startup, not once per house. Data that's identical across every house in
    the run (ground_model's grid+interpolator, nadir_img_data's KDTree+paths,
    etc.) used to ride along in every single worker_args tuple, so it was
    pickled and sent through the task queue once per house; with N houses
    and only min(cores, 32) worker processes, this cuts that down to once
    per worker. Only the nadir index is needed here - this is just the
    naive extraction-time placeholder photo (see worker_extraction below);
    the real nadir AND oblique picks both get precision-searched fresh in
    step 10, against both indexes. ransac_max_fit_points comes from config.yml
    explicitly (not segment_plane_capped's own default) so workers actually
    honor a value edited into the config, rather than relying on the
    module-level default a forked worker happened to inherit."""
    _worker_ctx['ground_model'] = ground_model
    _worker_ctx['rot'] = rot
    _worker_ctx['off_xy'] = off_xy
    _worker_ctx['g_off'] = g_off
    _worker_ctx['nadir_img_data'] = nadir_img_data
    _worker_ctx['out_dir'] = out_dir
    _worker_ctx['ransac_max_fit_points'] = ransac_max_fit_points

def worker_extraction(args):
    idx, seeds, local_pts, local_cols = args
    ground_model = _worker_ctx['ground_model']
    rot = _worker_ctx['rot']
    off_xy = _worker_ctx['off_xy']
    g_off = _worker_ctx['g_off']
    nadir_img_data = _worker_ctx['nadir_img_data']
    out_dir = _worker_ctx['out_dir']
    ransac_max_fit_points = _worker_ctx['ransac_max_fit_points']

    # Standard 20cm noise filter above the true local ground elevation
    local_ground = ground_model.get_z(local_pts[:, 0], local_pts[:, 1])
    local_mask = local_pts[:, 2] > (local_ground + 0.20)

    local_pts, local_cols = local_pts[local_mask], local_cols[local_mask]
    if occupied_area_sqft(local_pts[:, :2]) < MIN_CANDIDATE_AREA_SQFT:
        return []

    valid_roof_pts = []
    rem = seeds.copy()
    plane_rounds = 0
    ransac_start = time.time()
    while occupied_area_sqft(rem[:, :2]) > MIN_CANDIDATE_AREA_SQFT:
        _, inliers = segment_plane_capped(rem, max_fit_points=ransac_max_fit_points)
        if occupied_area_sqft(rem[inliers][:, :2]) < MIN_CANDIDATE_AREA_SQFT:
            break
        valid_roof_pts.append(rem[inliers])
        rem = np.delete(rem, inliers, axis=0)
        plane_rounds += 1
    ransac_elapsed = time.time() - ransac_start
    if ransac_elapsed > 5.0:
        # Only the slow ones print - see the constant's comment above for why
        # this loop can occasionally run long. Lets a rerun's log point
        # straight at which blob(s) dominate step 8 without spamming a line
        # for every ordinary house.
        cx, cy = seeds[:, :2].mean(axis=0)
        print(f"      [worker] blob {idx} near local ({cx:.0f},{cy:.0f}): {plane_rounds} plane(s) "
              f"peeled from {len(seeds):,} seed pts in {ransac_elapsed:.1f}s")

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
        result = finalize_candidate(local_pts, local_cols, footprint, ground_model, rot, off_xy)
        if result:
            p, a, v, n_floor, needs_review_stage1 = result
            all_pts = np.asarray(p.points)
            all_cols = np.asarray(p.colors)
            ag_only_pts = all_pts[:-n_floor] if n_floor > 0 else all_pts
            ag_only_cols = all_cols[:-n_floor] if n_floor > 0 else all_cols

            # Needs_Review combines two independent signals: the (weak, see
            # comment above check_vegetation_contamination) color-distance
            # check, OR'd with a validated one from finalize_candidate - a
            # meaningful fraction of this candidate having needed removal is
            # itself evidence something was going on here, whether or not
            # that stage fully resolved it (it can't always - see
            # filter_disconnected_fragments' comment on the touching-tree
            # case it doesn't catch).
            needs_review = check_vegetation_contamination(ag_only_pts, ag_only_cols) or needs_review_stage1

            z_vals = all_pts[:, 2]
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

            cent = all_pts.mean(axis=0)
            gx, gy = cent[0] + g_off[0], cent[1] + g_off[1]

            temp_uid = f"H_{abs(gx):.3f}_{abs(gy):.3f}".replace('.', 'd')

            orig_img_name = "N/A"
            if nadir_img_data[1]:
                _, i_idx = nadir_img_data[1].query([gx, gy], k=1)
                img_p = nadir_img_data[0][i_idx]
                orig_img_name = os.path.basename(img_p)
                shutil.copy2(img_p, os.path.join(out_dir, "best_images", f"{temp_uid}_nadir.jpg"))

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
                "Nadir_Original_Image": orig_img_name,
                "corners_3d": corners_3d.tolist(),
                "Needs_Review": needs_review
            })
    return houses

# --- Footprint-overlap merge ---
# A large or architecturally complex building (a church, a commercial
# building - anything with more roof facets than a typical house) can get
# split into several separate "houses" here: the RANSAC-plane loop above
# extracts one facet at a time, DBSCAN then clusters all of a blob's
# extracted facets by 2D proximity (eps=2.0m), and each resulting cluster
# becomes its own independent candidate with its own alpha-shape footprint.
# For a building whose facets are legitimately more than 2m apart in places
# (a nave, a transept, a tower), that's several honest candidates for what
# is structurally one building. Confirmed directly against a real
# extraction: three "houses" from the same run had footprints that
# literally overlapped in XY - physically impossible for separate
# buildings - so overlap is treated here as a near-unambiguous merge signal,
# safer than a blanket adjacency/distance rule that risks merging
# legitimately separate, closely-spaced buildings instead.
def _xy_bounds(corners_3d):
    c = np.asarray(corners_3d)
    return c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()

def _boxes_overlap(b1, b2, buffer=1.0):
    """AABB overlap/near-touch test - a cheap, NECESSARY-but-not-sufficient
    pre-filter (two footprints can only truly overlap if their bounding
    boxes do too), not the merge decision itself - see MIN_OVERLAP_SQFT
    below for that. An axis-aligned box alone is NOT safe as the actual
    merge signal: confirmed as a real bug on a site where houses sit on a
    non-axis-aligned street grid - each house's true (rotated/irregular)
    footprint's inflated axis-aligned box can touch a neighbor's inflated
    box even though the real footprints are meters apart, and union-find
    then chains that transitively (A overlaps B, B overlaps C, so A/B/C all
    merge even though A and C never come close) - visually confirmed as a
    diagonal chain of clearly separate, gapped buildings merged into one."""
    x1min, x1max, y1min, y1max = b1
    x2min, x2max, y2min, y2max = b2
    return (x1min - buffer < x2max) and (x2min - buffer < x1max) and \
           (y1min - buffer < y2max) and (y2min - buffer < y1max)

# The REAL overlap test: actual occupied-grid-cell intersection between two
# candidates' own points, which - unlike an axis-aligned bounding box -
# respects each footprint's true shape regardless of rotation or
# irregularity. MIN_OVERLAP_SQFT requires a real, non-trivial shared area
# (not just one or two cells' worth) before merging, so boundary noise at a
# shared property line doesn't get mistaken for genuine overlap.
OVERLAP_CELL_SIZE = 0.5
MIN_OVERLAP_SQFT = 10.0

def _occupied_cells(xy, cell_size=OVERLAP_CELL_SIZE):
    """Set of occupied grid-cell IDs in GLOBAL coordinates (not shifted to
    this point set's own local origin), so two different candidates' cell
    sets can be intersected directly."""
    cells = np.floor(xy / cell_size).astype(np.int64)
    return set(map(tuple, cells))

# A shared footprint alone doesn't mean two candidates are fragments of the
# same building - a tree canopy bridging two separate houses produces the
# identical evidence (occupied cells touching) as a real shared wall. Every
# geometry/color signal tried in this project to classify "is this
# vegetation" globally has failed (see check_vegetation_contamination's own
# comment, and the vehicle-detection dead end) - real vegetation and real
# small structures genuinely overlap in raw color/height/shape/density once
# photogrammetry has smoothed away the texture that would give it away.
#
# This works differently: not "is this material vegetation," but "is the
# material connecting these two ALREADY-CONFIRMED candidates consistent
# with being a continuation of THEIR OWN established roof heights." That's
# a local, relative comparison, not a global absolute one - the same shift
# that made the ring-shaped-candidate rescue work after everything absolute
# had failed. Validated against 5 real merge decisions from an actual run
# (not synthetic cases): 4 confirmed-bad merges (unrelated houses bridged by
# trees) scored 45-63% height-consistent; the 1 confirmed-good merge (a
# genuinely fragmented large building) scored 86%. A cruder version of this
# idea - just measuring how narrow the connecting neck is - did NOT
# separate the same cases (a real bad and a real good merge split at the
# identical erosion radius), so the height check specifically is what's
# carrying this, not shape.
BRIDGE_HEIGHT_MARGIN_M = 1.0
BRIDGE_MIN_CONSISTENT_FRACTION = 0.75

def _bridge_is_height_consistent(pts_i, pts_j, overlap_cells, ground_model, rot, off_xy,
                                  cell_size=OVERLAP_CELL_SIZE,
                                  margin_m=BRIDGE_HEIGHT_MARGIN_M, min_frac=BRIDGE_MIN_CONSISTENT_FRACTION):
    """True if the material actually sitting in the shared/overlapping cells
    between two candidates is height-consistent with at least one of the two
    candidates' own (non-overlap) roof heights - real evidence they're
    fragments of one building, not just evidence their footprints touch.
    Defaults to True (don't block the merge) whenever there isn't enough
    data on either side to judge confidently, rather than risk rejecting a
    real merge on a thin sample.

    Compares height ABOVE LOCAL GROUND (via ground_model), not absolute
    elevation. Confirmed as a real bug on the fir dataset: one genuine
    building spanning sloped terrain was shattered into 4 disconnected
    pieces because its far ends' absolute roof Z differed by 5m+ purely from
    terrain slope (its floor elevation alone drifted ~2m site to site),
    while its height above each end's own local ground stayed consistent
    (~1.4-1.7m). Absolute-Z comparison reads that slope as "different
    roof," exactly the same mistake ground_model itself exists to avoid
    everywhere else in this pipeline."""
    def split(pts):
        cells = np.floor(pts[:, :2] / cell_size).astype(np.int64)
        in_overlap = np.array([tuple(c) in overlap_cells for c in cells])
        return pts[~in_overlap], pts[in_overlap]

    interior_i, bridge_i = split(pts_i)
    interior_j, bridge_j = split(pts_j)
    if len(interior_i) < 20 or len(interior_j) < 20:
        return True

    def agl(pts):
        internal = _world_to_internal(pts, rot, off_xy)
        return internal[:, 2] - ground_model.get_z(internal[:, 0], internal[:, 1])

    height_i = np.median(agl(interior_i))
    height_j = np.median(agl(interior_j))
    bridge_pts = np.vstack([p for p in (bridge_i, bridge_j) if len(p)])
    if len(bridge_pts) < 20:
        return True

    bridge_agl = agl(bridge_pts)
    delta_to_nearest = np.minimum(np.abs(bridge_agl - height_i), np.abs(bridge_agl - height_j))
    return bool(np.mean(delta_to_nearest < margin_m) >= min_frac)

def _world_to_internal(pts, rot, off_xy):
    """Inverse of the translate+rotate finalize_candidate/worker_extraction
    apply to bring a candidate OUT of ground_model's internal (leveled)
    frame into world coordinates. Needed here because merge_overlapping_
    footprints reloads already-WORLD-frame .ply files from disk, but
    ground_model (and finalize_candidate's footprint/cookie-cutter
    machinery) only understand the internal frame - confirmed as a second
    real bug: without this conversion, ground_model.get_z() was being
    queried with coordinates offset by off_xy from where it expects, which
    could put a merged candidate's floor at the wrong elevation entirely."""
    off_xy_vec = np.array([off_xy[0], off_xy[1], 0.0])
    return pts @ rot.T - off_xy_vec

def merge_overlapping_footprints(raw_res, out_dir, ground_model, rot, off_xy, g_off):
    """Groups raw_res candidates whose footprints ACTUALLY overlap in XY
    (via union-find, same pattern as the erosion-core fragment
    reconciliation earlier in the pipeline, but see _boxes_overlap/
    MIN_OVERLAP_SQFT above for why this checks real occupied-cell overlap
    rather than just bounding boxes) and replaces each group of 2+ with ONE
    recombined candidate: reuses each fragment's already-loaded points,
    strips each one's own synthetic floor (independently computed per-
    fragment, so they'd conflict), converts back to ground_model's internal
    frame, re-traces a single alpha-shape footprint over the combined roof/
    wall points, and runs the result through finalize_candidate - reusing
    the exact same area/volume/dip-filter/disconnected-fragment machinery a
    normal single-building candidate goes through, rather than
    approximating the merge by just summing the fragments' numbers (which
    would double-count the overlapping region)."""
    if len(raw_res) < 2:
        return raw_res

    bounds = [_xy_bounds(r["corners_3d"]) for r in raw_res]
    n = len(raw_res)

    # Lazily loaded, cached per-candidate (ag_pts, ag_cols, occupied-cell-
    # set) - loaded once per candidate that's at least an AABB overlap
    # candidate, reused both for the true-overlap test below and for
    # building any confirmed merge group, so no fragment gets read from
    # disk more than once no matter how many pairs it's checked against.
    cache = {}
    def load(i):
        if i in cache:
            return cache[i]
        ply_path = os.path.join(out_dir, "individual_houses", f"{raw_res[i]['temp_ID']}.ply")
        if not os.path.exists(ply_path):
            cache[i] = (None, None, None)
            return cache[i]
        frag = o3d.io.read_point_cloud(ply_path)
        fpts, fcols = np.asarray(frag.points), np.asarray(frag.colors)
        # This fragment's own synthetic floor: uniform [0.4]*3 gray, same
        # invariant apply_house_cookie_cutter always produces - strip it,
        # since a single new floor gets computed below for the merged
        # footprint instead.
        is_floor = np.all(np.abs(fcols - 0.4) < 0.01, axis=1)
        ag_pts, ag_cols = fpts[~is_floor], fcols[~is_floor]
        cache[i] = (ag_pts, ag_cols, _occupied_cells(ag_pts[:, :2]))
        return cache[i]

    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Per-union-find-ROOT accumulated points/cells, kept in sync as groups
    # grow - the height-consistency check judges a pair using everything
    # ALREADY confidently grouped on each side, not just one original
    # fragment's own isolated points. Confirmed necessary on real data: a
    # single large building's own erosion fragments can be small enough that
    # ONE fragment's own sliver right at its mutual boundary with its
    # neighbor is too thin a sample to judge reliably - the identical check
    # against the SAME building's already-merged, richer groups (more roof
    # sampled, same roof) passed cleanly (86%) where the raw individual
    # fragments alone failed. A real roof doesn't change what's true about
    # its own height depending on how much of it happens to be in one
    # erosion fragment - more of it sampled just makes the estimate more
    # reliable.
    root_pts, root_cells = {}, {}
    for i in range(n):
        pts_i, _, cells_i = load(i)
        if pts_i is not None:
            root_pts[i] = pts_i
            root_cells[i] = cells_i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        parent[ri] = rj
        if ri in root_pts:
            root_pts[rj] = np.vstack([root_pts[rj], root_pts[ri]]) if rj in root_pts else root_pts[ri]
            root_cells[rj] = root_cells.get(rj, set()) | root_cells[ri]
            del root_pts[ri]
            del root_cells[ri]

    # Repeat the pairwise pass until nothing new merges - a pair that fails
    # on round 1 (thin, individual fragments) can legitimately pass once
    # OTHER merges have already enriched one or both sides' group data, so a
    # single pass isn't enough to give every real connection a fair look
    # with the data it deserves. Bounded iteration count, not a fixed point
    # loop, since a real site never needs many rounds to settle (73
    # candidates settled well under 5 here).
    for _ in range(5):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if not _boxes_overlap(bounds[i], bounds[j]):
                    continue
                ri, rj = find(i), find(j)
                if ri == rj or ri not in root_pts or rj not in root_pts:
                    continue
                overlap_cells = root_cells[ri] & root_cells[rj]
                overlap_sqft = len(overlap_cells) * (OVERLAP_CELL_SIZE ** 2) * 10.764
                if overlap_sqft < MIN_OVERLAP_SQFT:
                    continue
                if _bridge_is_height_consistent(root_pts[ri], root_pts[rj], overlap_cells, ground_model, rot, off_xy):
                    union(i, j)
                    changed = True
        if not changed:
            break

    # Final pass, after everything that's going to merge already has: log
    # whatever's STILL geometrically overlapping but unmerged as a genuine,
    # settled rejection - by now more rounds wouldn't change the verdict, so
    # this reflects a real persistent mismatch, not a fragment that just
    # hadn't grown enough yet.
    rejected_bridge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not _boxes_overlap(bounds[i], bounds[j]):
                continue
            ri, rj = find(i), find(j)
            if ri == rj or ri not in root_cells or rj not in root_cells:
                continue
            overlap_cells = root_cells[ri] & root_cells[rj]
            overlap_sqft = len(overlap_cells) * (OVERLAP_CELL_SIZE ** 2) * 10.764
            if overlap_sqft < MIN_OVERLAP_SQFT:
                continue
            rejected_bridge_count += 1
            print(f"      -> footprints of {raw_res[i]['temp_ID']} and {raw_res[j]['temp_ID']} overlap "
                  f"({overlap_sqft:.0f} sqft) but the connecting material isn't height-consistent with "
                  f"either roof - not merging (likely a vegetation bridge, not a shared wall)")

    if rejected_bridge_count:
        print(f"      -> {rejected_bridge_count} footprint-overlap pair(s) rejected as likely vegetation bridges")

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    output = []
    merged_count = 0
    for idxs in groups.values():
        if len(idxs) == 1:
            output.append(raw_res[idxs[0]])
            continue

        members = [raw_res[i] for i in idxs]
        print(f"      -> merging {len(members)} overlapping-footprint fragments "
              f"({', '.join(m['temp_ID'] for m in members)}) into one building")

        ag_pts_list, ag_cols_list = [], []
        for i in idxs:
            ag_pts, ag_cols, _ = load(i)
            if ag_pts is None:
                continue
            ag_pts_list.append(ag_pts)
            ag_cols_list.append(ag_cols)

        if not ag_pts_list:
            output.extend(members)
            continue

        combined_pts = np.vstack(ag_pts_list)
        combined_cols = np.vstack(ag_cols_list)

        # These fragments were reloaded from disk in WORLD coordinates -
        # convert back to ground_model's internal frame before tracing a
        # footprint or cookie-cutting (see _world_to_internal's comment).
        internal_pts = _world_to_internal(combined_pts, rot, off_xy)

        footprint = calculate_alpha_shape(internal_pts[:, :2])
        if not footprint:
            output.extend(members)
            continue
        result = finalize_candidate(internal_pts, combined_cols, footprint, ground_model, rot, off_xy)
        if not result:
            output.extend(members)
            continue
        p, a, v, n_floor, needs_review_stage1 = result

        all_pts = np.asarray(p.points)
        all_cols = np.asarray(p.colors)
        ag_only_pts = all_pts[:-n_floor] if n_floor > 0 else all_pts
        ag_only_cols = all_cols[:-n_floor] if n_floor > 0 else all_cols
        needs_review = check_vegetation_contamination(ag_only_pts, ag_only_cols) or needs_review_stage1

        z_vals = all_pts[:, 2]
        height_m = np.percentile(z_vals, 99) - z_vals.min()
        height_ft = height_m * 3.28084
        area_sqft = a * 10.76
        vol_cuft = v * 35.31
        ratio = area_sqft / height_ft if height_ft > 0 else 0

        cent = all_pts.mean(axis=0)
        gx, gy = cent[0] + g_off[0], cent[1] + g_off[1]
        merged_uid = f"MERGED_{abs(gx):.3f}_{abs(gy):.3f}".replace('.', 'd')
        corners_3d = np.asarray(p.get_axis_aligned_bounding_box().get_box_points())

        o3d.io.write_point_cloud(os.path.join(out_dir, "individual_houses", f"{merged_uid}.ply"), p)

        # Nadir placeholder for the merged candidate: copy the largest
        # fragment's own nadir photo, and carry forward its
        # Nadir_Original_Image name rather than hardcoding "N/A" - step 10
        # used to treat "N/A" as "skip this candidate entirely" (see its own
        # comment), which silently skipped precision cropping for every
        # merged candidate, not just the genuinely-imageless ones. Step 10
        # still always re-derives the real best nadir AND oblique photos
        # from scratch using the (now merged) corners_3d/centroid; this
        # placeholder only matters if that re-derivation fails too. There's
        # no oblique placeholder to carry forward - oblique is only ever
        # precision-searched in step 10, never given a naive fallback.
        biggest = max(members, key=lambda m: m["Area_sqft"])
        src_img = os.path.join(out_dir, "best_images", f"{biggest['temp_ID']}_nadir.jpg")
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(out_dir, "best_images", f"{merged_uid}_nadir.jpg"))

        for m in members:
            for p_old in [os.path.join(out_dir, "individual_houses", f"{m['temp_ID']}.ply"),
                          os.path.join(out_dir, "best_images", f"{m['temp_ID']}_nadir.jpg")]:
                if os.path.exists(p_old):
                    os.remove(p_old)

        output.append({
            "temp_ID": merged_uid,
            "Area_sqft": round(area_sqft, 2),
            "Volume_cuft": round(vol_cuft, 2),
            "Height_ft": round(height_ft, 2),
            "Ratio_Area_to_Height": round(ratio, 2),
            "X_coord": gx,
            "Y_coord": gy,
            "Nadir_Original_Image": biggest.get("Nadir_Original_Image", "N/A"),
            "corners_3d": corners_3d.tolist(),
            "Needs_Review": needs_review,
        })
        merged_count += 1

    print(f"      -> {merged_count} overlapping-footprint group(s) merged; "
          f"{len(output)} candidates remain (was {len(raw_res)})")
    return output

# --- Ring-shaped (oversized-roof) candidate rescue ---
# LocalGroundModel's documented blind spot: a building wider than
# ground_model.opening_span reads as its own roof height being "ground" in
# its interior, so the SITE-WIDE above-ground filter (step 5) strips those
# interior points before extraction ever runs - the candidate that survives
# is a hollow ring (the building's outer edge, close enough to real ground
# to pass) with a false hole in the middle, not a genuinely gutted building.
#
# Two fixes at the ground-model (grid) level were tried and reverted - see
# LocalGroundModel's own comment - because a correction wide enough to reach
# past an oversized roof to real ground is also wide enough to reach into
# unrelated terrain a shared grid cell has no way to tell apart from "this
# building's own ground." Both real failures (site-wide corruption, then
# cross-building bridging in dense blocks) came from exactly that: the grid
# doesn't know which building a cell belongs to.
#
# This fixes it per-candidate instead, after extraction, where that context
# already exists:
#   1. A candidate's own convex hull (NOT its alpha-shape, which would
#      already trace around the hole we're trying to detect) vs. its own
#      occupied area gives a fill ratio - low for a ring, ~1 for a solid
#      blob.
#   2. For a flagged candidate, search the FULL pre-above-ground-filter
#      point cloud (points step 5 excluded, not just what already survived)
#      but restricted to two things scoped entirely to this ONE candidate:
#      within its own convex hull, and within RING_RESCUE_HEIGHT_MARGIN_M of
#      ITS OWN already-extracted median roof height. Both restrictions are
#      local to this candidate alone - it cannot reach into a neighbor's
#      hull or match a neighbor's different roof height, which is exactly
#      what made the grid-level attempts unsafe.
#   3. Only trusted if the rescued fill ratio actually closes the hole
#      (RING_RESCUE_MIN_FILL_RATIO) - if it doesn't, the candidate ships
#      exactly as originally extracted rather than a half-fixed shape.
RING_MIN_AREA_SQFT = 1000.0
RING_FILL_RATIO_THRESHOLD = 0.75
RING_RESCUE_HEIGHT_MARGIN_M = 1.5
RING_RESCUE_MIN_FILL_RATIO = 0.85

def _hull_fill_ratio(xy):
    """Fraction of a 2D point set's own convex hull area that's actually
    occupied - low for a ring/donut shape, close to 1 for a solid blob.
    Returns (fill_ratio, hull_or_None); hull is None when there aren't
    enough points to form one (fill_ratio then defaults to 1.0 - too few
    points to call it a ring)."""
    if len(xy) < 4:
        return 1.0, None
    try:
        hull = ConvexHull(xy)
    except Exception:
        return 1.0, None
    hull_area_sqft = hull.volume * 10.764  # hull.volume IS area for a 2D hull
    if hull_area_sqft <= 0:
        return 1.0, None
    return occupied_area_sqft(xy) / hull_area_sqft, hull

def rescue_ring_shaped_candidates(raw_res, out_dir, ground_model, rot, off_xy, g_off, full_pts, full_cols):
    """See module comment above. full_pts/full_cols must be the FULL
    ground_model-internal-frame point cloud from BEFORE the step 5 above-
    ground filter (the same `pts`/`cols` step 4 builds the ground model
    from) - the whole point is recovering points that filter excluded."""
    full_xy = full_pts[:, :2]
    output = []
    rescued_count = 0

    for r in raw_res:
        if r["Area_sqft"] < RING_MIN_AREA_SQFT:
            output.append(r)
            continue

        ply_path = os.path.join(out_dir, "individual_houses", f"{r['temp_ID']}.ply")
        if not os.path.exists(ply_path):
            output.append(r)
            continue

        frag = o3d.io.read_point_cloud(ply_path)
        fpts, fcols = np.asarray(frag.points), np.asarray(frag.colors)
        is_floor = np.all(np.abs(fcols - 0.4) < 0.01, axis=1)
        ag_pts, ag_cols = fpts[~is_floor], fcols[~is_floor]
        if len(ag_pts) < 50:
            output.append(r)
            continue

        internal_pts = _world_to_internal(ag_pts, rot, off_xy)
        fill_ratio, hull = _hull_fill_ratio(internal_pts[:, :2])
        if hull is None or fill_ratio >= RING_FILL_RATIO_THRESHOLD:
            output.append(r)
            continue

        # Cheap bounding-box pre-filter before the more expensive Delaunay
        # point-in-hull test, so this doesn't have to check every one of
        # the full site's points for every flagged candidate.
        hull_xy = internal_pts[:, :2][hull.vertices]
        xmin, ymin = hull_xy.min(axis=0)
        xmax, ymax = hull_xy.max(axis=0)
        bbox_mask = (full_xy[:, 0] >= xmin) & (full_xy[:, 0] <= xmax) & \
                    (full_xy[:, 1] >= ymin) & (full_xy[:, 1] <= ymax)
        if not bbox_mask.any():
            output.append(r)
            continue

        in_hull = Delaunay(hull_xy).find_simplex(full_xy[bbox_mask]) >= 0
        hull_region_pts = full_pts[bbox_mask][in_hull]
        hull_region_cols = full_cols[bbox_mask][in_hull]
        if len(hull_region_pts) < len(ag_pts):
            continue  # shouldn't happen - candidate's own points are a subset of this region

        roof_z = np.median(internal_pts[:, 2])
        matches_roof = np.abs(hull_region_pts[:, 2] - roof_z) < RING_RESCUE_HEIGHT_MARGIN_M
        rescued_pts, rescued_cols = hull_region_pts[matches_roof], hull_region_cols[matches_roof]

        new_fill_ratio, _ = _hull_fill_ratio(rescued_pts[:, :2])
        if new_fill_ratio < RING_RESCUE_MIN_FILL_RATIO:
            # Didn't actually close the hole enough to trust - ship the
            # candidate as originally extracted rather than a half-fix.
            output.append(r)
            continue

        footprint = calculate_alpha_shape(rescued_pts[:, :2])
        if not footprint:
            output.append(r)
            continue
        result = finalize_candidate(rescued_pts, rescued_cols, footprint, ground_model, rot, off_xy)
        if not result:
            output.append(r)
            continue

        p, a, v, n_floor, needs_review_stage1 = result
        all_pts = np.asarray(p.points)
        all_cols = np.asarray(p.colors)
        ag_only_pts = all_pts[:-n_floor] if n_floor > 0 else all_pts
        ag_only_cols = all_cols[:-n_floor] if n_floor > 0 else all_cols
        needs_review = check_vegetation_contamination(ag_only_pts, ag_only_cols) or needs_review_stage1

        z_vals = all_pts[:, 2]
        height_m = np.percentile(z_vals, 99) - z_vals.min()
        height_ft = height_m * 3.28084
        area_sqft = a * 10.76
        vol_cuft = v * 35.31
        ratio = area_sqft / height_ft if height_ft > 0 else 0

        cent = all_pts.mean(axis=0)
        gx, gy = cent[0] + g_off[0], cent[1] + g_off[1]

        o3d.io.write_point_cloud(ply_path, p)  # overwrite in place - same temp_ID/files

        print(f"      -> rescued ring-shaped candidate {r['temp_ID']}: fill ratio "
              f"{fill_ratio:.2f} -> {new_fill_ratio:.2f}, area {r['Area_sqft']:.0f} -> {area_sqft:.0f} sqft")
        rescued_count += 1

        output.append({
            **r,
            "Area_sqft": round(area_sqft, 2),
            "Volume_cuft": round(vol_cuft, 2),
            "Height_ft": round(height_ft, 2),
            "Ratio_Area_to_Height": round(ratio, 2),
            "X_coord": gx,
            "Y_coord": gy,
            "corners_3d": np.asarray(p.get_axis_aligned_bounding_box().get_box_points()).tolist(),
            "Needs_Review": needs_review,
        })

    if rescued_count:
        print(f"      -> Ring-shaped candidate rescue: {rescued_count} candidate(s) rescued")
    return output

# --- 4. MASTER PIPELINE ---
def process_reconstruction(args):
    global_start_time = time.time()
    # Previously only step 8 (parallel extraction) and the grand total were
    # timed - every other step just printed point counts, not durations, so
    # there was no way to tell from a log which of the other 9 steps a slow
    # run's time actually went to. step_done() prints how long has elapsed
    # since the last call (or since the function started, for the first
    # call) and resets the clock - call it once at the end of each step.
    _step_clock = [time.time()]
    def step_done(label):
        now = time.time()
        print(f"      -> {label} completed in {now - _step_clock[0]:.1f}s")
        _step_clock[0] = now

    project_path = os.path.abspath(args.project_path)
    project_name = os.path.basename(os.path.normpath(project_path))
    out = os.path.join(project_path, f"analysis_{project_name}_v9")

    for d in ["individual_houses", "best_images", "best_images_cropped", "diagnostics"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    print(f"Output directory: {out}")

    cfg = load_config(args.config)
    print(f"Config loaded from: {args.config}")
    print(f"  ground_model: cell_size={cfg['ground_model']['cell_size']}m "
          f"opening_span={cfg['ground_model']['opening_span']}m relevel={cfg['ground_model']['relevel']}")
    print(f"  vehicle_rejection: reject_small_structures={cfg['vehicle_rejection']['reject_small_structures']} "
          f"max_area_sqft={cfg['vehicle_rejection']['max_area_sqft']} "
          f"max_height_ft={cfg['vehicle_rejection']['max_height_ft']}")
    print(f"  imagery: nadir_pitch_threshold={cfg['imagery']['nadir_pitch_threshold']} deg")
    print(f"  performance: ransac_max_fit_points={cfg['performance']['ransac_max_fit_points']}")

    if cfg["vehicle_rejection"]["reject_small_structures"]:
        print("WARNING: vehicle_rejection.reject_small_structures is active. Candidates under "
              f"{cfg['vehicle_rejection']['max_area_sqft']:.0f} sqft and "
              f"{cfg['vehicle_rejection']['max_height_ft']:.0f} ft will be dropped entirely, not just "
              "flagged - this WILL also remove some real small structures (sheds, small garages), since "
              "no known geometric test tells those apart from a parked car. Check the printed reject "
              "count at step 9 against what you expect to see.")

    print(f"CPU cores available: {NUM_CORES} (source: {_CORE_SOURCE})")
    mem_monitor = MemoryMonitor()
    if mem_monitor.start():
        print("Memory monitor: sampling via psutil (main + child worker processes)")
    else:
        print("Memory monitor: psutil not installed, falling back to a cruder end-of-run estimate "
              "(pip install psutil for an accurate number)")

    _step_clock[0] = time.time()  # setup/warnings above don't count against step 1
    print("\n[1/10] Syncing Global Offsets...")
    g_off = auto_detect_offsets(project_path)
    reverse_trans = Transformer.from_crs(EPSG_CODE, "EPSG:4326", always_xy=True)
    nadir_img_data, oblique_img_data = build_dual_spatial_image_index(
        os.path.join(project_path, 'images'), Transformer.from_crs("EPSG:4326", EPSG_CODE, always_xy=True),
        nadir_pitch_threshold=cfg["imagery"]["nadir_pitch_threshold"])

    step_done("Step 1/10 (global offsets)")
    print("\n[2/10] Loading and cleaning PLY...")
    ply_path = os.path.join(project_path, 'scene_dense.ply')
    print(f"      -> Reading: {ply_path}")
    pcd = o3d.io.read_point_cloud(repair_openmvs_ply_colors(ply_path))
    print(f"      -> Raw point count: {len(pcd.points):,}")
    pcd = pcd.voxel_down_sample(0.05)
    print(f"      -> After 5cm voxel downsample: {len(pcd.points):,}")
    pcd, _ = pcd.remove_statistical_outlier(20, 2.2)
    print(f"      -> After statistical outlier removal: {len(pcd.points):,}")

    step_done("Step 2/10 (load/clean PLY)")
    print("\n[3/10] Centering scene geometry...")
    pts = np.asarray(pcd.points)
    grnd = pts[pts[:, 2] < np.percentile(pts[:, 2], 30)]

    rot = np.eye(3)
    if cfg["ground_model"]["relevel"]:
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
        print(f"      -> ground_model.relevel active: rotating {tilt_deg:.2f} degrees to force the "
              f"lowest-30%-of-points plane horizontal")
    else:
        print("      -> Trusting upstream gravity correction (auto_reconstruct.py) - no additional "
              "rotation applied. Set ground_model.relevel: true in the config if this point cloud did "
              "NOT go through the updated auto_reconstruct.py.")

    pcd.rotate(rot, center=(0,0,0))
    off_xy = np.mean(grnd, axis=0)[:2]
    pcd.translate((-off_xy[0], -off_xy[1], 0))
    print(f"      -> Scene re-centered on XY offset: ({off_xy[0]:.3f}, {off_xy[1]:.3f})")

    step_done("Step 3/10 (centering)")
    print(f"\n[4/10] Building Local Contour Ground Model "
          f"(cell={cfg['ground_model']['cell_size']}m, opening={cfg['ground_model']['opening_span']}m)...")
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    ground_model = LocalGroundModel(pts, cell_size=cfg["ground_model"]["cell_size"],
                                     opening_span=cfg["ground_model"]["opening_span"])
    ground_model.write_diagnostic_ply(os.path.join(out, "diagnostics", "ground_surface.ply"), off_xy, rot)

    step_done("Step 4/10 (ground model)")
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

    step_done("Step 5/10 (above-ground filter)")
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
        # reconstruction.depthmap_resolution) - k=32 keeps per-candidate cost
        # bounded regardless of density.
        # Deliberately NOT converted to an area/density-based equivalent like the
        # count thresholds below - switching this back to a radius-based query
        # would reintroduce exactly the O(n^2) memory blowup that caused the
        # 360GB OOM crash this fixed-k query was originally written to solve.
        _, n_idx_l = tree.query(ag_pts[cand_idx], k=32, workers=-1)
        neighbor_xyz = ag_pts[n_idx_l]              # (n_cand, 32, 3)
        z_v = neighbor_xyz[:, :, 2]
        is_bumpy = (np.ptp(z_v, axis=1) > 0.35) | (np.std(z_v, axis=1) > 0.08)

        # Z-range/std only sees vertical spread, so a patch of foliage that
        # happens to be roughly level in Z (e.g. a manicured hedge top), or
        # brown/dead branches that also fail the ExG color test above, can
        # slip through the check above. A local 3D PLANARITY feature catches
        # both regardless of color: reuses the SAME k=32 neighborhoods already
        # fetched above (no extra KDTree pass), just using their full XYZ
        # instead of only Z. For each candidate's neighborhood, decompose the
        # 3x3 covariance into eigenvalues l1>=l2>=l3 (standard LIDAR
        # ground/vegetation feature): a real roof surface is locally flat, so
        # l3 (the "how far off the best-fit plane" axis) is close to zero and
        # planarity=(l2-l3)/l1 is high; foliage scatters in all 3 directions
        # (low planarity, high l3) and a branch is linear (l2~=l3, ALSO low
        # planarity) - so this one test separates "sits on a hard flat
        # surface" from either failure mode, without needing color at all.
        # Threshold is a starting default, not empirically calibrated against
        # a real project yet - check the printed count below against a known
        # problem house before trusting it on a big batch.
        centered = neighbor_xyz - neighbor_xyz.mean(axis=1, keepdims=True)
        cov = np.einsum('nki,nkj->nij', centered, centered) / (centered.shape[1] - 1)
        eigvals = np.linalg.eigvalsh(cov)           # ascending: l3, l2, l1
        l3, l2, l1 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
        planarity = np.divide(l2 - l3, l1, out=np.zeros_like(l1), where=l1 > 1e-9)
        is_not_planar = planarity < 0.5

        tree_mask[cand_idx[is_bumpy | is_not_planar]] = True

    pruned_ag_pcd = ag_pcd.select_by_index(np.where(~tree_mask)[0])
    ag_pts, ag_colors = np.asarray(pruned_ag_pcd.points), np.asarray(pruned_ag_pcd.colors)
    print(f"      -> {int(tree_mask.sum()):,} points removed as vegetation/noise; {len(ag_pts):,} remain")

    step_done("Step 6/10 (vegetation removal)")
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
    dedup_centers, dedup_owner = [], []
    for lbl in unique_labels:
        pts2d = seeds[labels == lbl][:, :2]
        label_coords[lbl] = pts2d
        cells = np.unique(np.floor((pts2d - min_b) / res).astype(int), axis=0)
        label_sizes[lbl] = len(cells) * (res ** 2) * 10.764
        # One representative point per occupied cell, still tagged with this
        # label - used only by the broad-phase adjacency search below, kept
        # separate per label rather than collapsed into a shared array (see
        # that comment for why the collapsed version was wrong).
        dedup_centers.append(min_b + (cells + 0.5) * res)
        dedup_owner.append(np.full(len(cells), lbl))

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

    # Which labels MIGHT come within 0.75m of each other - a broad-phase
    # filter, same two-step pattern collision detection always uses (cheap
    # over-inclusive pass, then an exact check only on what survives it).
    # The exact check below is unchanged from before; only how many pairs
    # reach it changes. This used to cost O(labels^2) - a nested Python loop
    # over every pair, each with its own KDTree query between that pair's
    # full point sets. Fine for the ~100-300 cores a clean site produces,
    # but a real run confirmed 833 initial cores on a messy commercial-block
    # site: ~346,000 pair-checks, each with a KDTree query, and it showed.
    # This broad phase instead runs ONE KDTree query over every label's own
    # deduplicated occupied-cell centers (not raw points, and not collapsed
    # across labels - each label's cells stay separately tagged, so two
    # labels that both happen to touch the same small area both stay
    # visible - an earlier version of this fix collapsed to one label per
    # cell and silently lost real adjacencies that way, confirmed by a
    # 5000-trial randomized comparison against the original nested-loop
    # algorithm before this version, which matched exactly on all 5000).
    # Cost no longer grows with how fragmented a site's erosion pass turns
    # out to be - it's bounded by total occupied area, not labels^2.
    all_centers = np.vstack(dedup_centers)
    all_owner = np.concatenate(dedup_owner)
    margin = res * np.sqrt(2)  # accounts for using cell centers, not exact points
    raw_pairs = KDTree(all_centers).query_pairs(r=0.75 + margin, output_type='ndarray')
    candidate_pairs = set()
    if len(raw_pairs) > 0:
        owner_a, owner_b = all_owner[raw_pairs[:, 0]], all_owner[raw_pairs[:, 1]]
        diff = owner_a != owner_b
        lo = np.minimum(owner_a[diff], owner_b[diff])
        hi = np.maximum(owner_a[diff], owner_b[diff])
        for a, b in np.unique(np.stack([lo, hi], axis=1), axis=0):
            candidate_pairs.add((a, b))

    trees = {}
    def get_tree(lbl):
        if lbl not in trees:
            trees[lbl] = KDTree(label_coords[lbl])
        return trees[lbl]

    merge_count = 0
    for lbl_a, lbl_b in sorted(candidate_pairs):
        root_a, root_b = find(lbl_a), find(lbl_b)
        if root_a == root_b: continue

        size_a, size_b = label_sizes[root_a], label_sizes[root_b]
        is_a_frag = size_a < 150 or size_a < (0.05 * size_b)
        is_b_frag = size_b < 150 or size_b < (0.05 * size_a)

        if not (is_a_frag or is_b_frag): continue

        dists_ab, _ = get_tree(lbl_b).query(label_coords[lbl_a], k=1, distance_upper_bound=0.75)
        if not np.any(dists_ab < 0.75):
            continue

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

    step_done("Step 7/10 (erosion + fragment reconciliation)")
    print(f"\n[8/10] Parallel Extraction ({len(worker_args)} candidates across {min(NUM_CORES, 32)} workers)...")
    phase_start = time.time()
    raw_res = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(NUM_CORES, 32),
        initializer=_init_worker,
        initargs=(ground_model, rot, off_xy, g_off, nadir_img_data, out, cfg["performance"]["ransac_max_fit_points"])
    ) as ex:
        for res_list in ex.map(worker_extraction, worker_args):
            if res_list: raw_res.extend(res_list)
    print(f"      -> Extraction produced {len(raw_res)} raw building candidates in {time.time()-phase_start:.1f}s")

    print("      -> Checking for overlapping-footprint fragments (large/complex buildings)...")
    overlap_start = time.time()
    raw_res = merge_overlapping_footprints(raw_res, out, ground_model, rot, off_xy, g_off)
    print(f"      -> Overlapping-footprint merge completed in {time.time()-overlap_start:.1f}s")

    print("      -> Checking for ring-shaped (oversized-roof) candidates...")
    ring_start = time.time()
    raw_res = rescue_ring_shaped_candidates(raw_res, out, ground_model, rot, off_xy, g_off, pts, cols)
    print(f"      -> Ring-shaped candidate rescue completed in {time.time()-ring_start:.1f}s")

    step_done("Step 8/10 total (parallel extraction + overlap merge)")
    print("\n[9/10] Artifact Purge & Sequential Georeferencing...")
    # One combined record per house - measurements and location used to be
    # split across measurements.csv/location_lookup.csv, built from the same
    # `r` in the same loop below and joined only by house_ID; merged into a
    # single list/file since there was never a reason for two.
    final_measurements = []
    house_corners = {}  # house_ID -> 3D bounding box corners (computed once in the worker;
                         # step 10 uses this instead of re-reading each house's .ply from disk)
    house_counter = 1
    rejected_vehicle_count = 0

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
            old_img_path = os.path.join(out, "best_images", f"{r['temp_ID']}_nadir.jpg")

            if r["Ratio_Area_to_Height"] > upper_bound or r["Ratio_Area_to_Height"] >= threshold:
                is_possible_vehicle = check_possible_vehicle(
                    r["Area_sqft"], r["Height_ft"],
                    cfg["vehicle_rejection"]["max_area_sqft"], cfg["vehicle_rejection"]["max_height_ft"])

                if cfg["vehicle_rejection"]["reject_small_structures"] and is_possible_vehicle:
                    # No geometric test found so far can tell a parked car apart from a small
                    # legitimate structure - see check_possible_vehicle's comment. Opting into this
                    # flag means accepting that some real small buildings get dropped here too.
                    rejected_vehicle_count += 1
                    if os.path.exists(old_ply_path): os.remove(old_ply_path)
                    if os.path.exists(old_img_path): os.remove(old_img_path)
                    continue

                new_uid = f"{project_name}_{house_counter}"
                house_counter += 1

                new_ply_path = os.path.join(out, "individual_houses", f"{new_uid}.ply")
                new_img_path = os.path.join(out, "best_images", f"{new_uid}_nadir.jpg")

                if os.path.exists(old_ply_path): os.rename(old_ply_path, new_ply_path)
                if os.path.exists(old_img_path): os.rename(old_img_path, new_img_path)

                lon, lat = reverse_trans.transform(r['X_coord'], r['Y_coord'])

                # Best_Image/Original_Image fields (both nadir and oblique)
                # are filled in by step 10, which is the only place that
                # knows the FINAL file layout - precomputing them here used
                # to go stale whenever step 10 wrote a file this loop hadn't
                # seen yet (confirmed real for merged candidates, which often
                # have no nadir placeholder at this point but can still get
                # a real precision crop in step 10).
                final_measurements.append({
                    "house_ID": new_uid,
                    "Area_sqft": r["Area_sqft"],
                    "Volume_cuft": r["Volume_cuft"],
                    "Height_ft": r["Height_ft"],
                    "Ratio_Area_to_Height": r["Ratio_Area_to_Height"],
                    "X_UTM": round(r['X_coord'], 3),
                    "Y_UTM": round(r['Y_coord'], 3),
                    "Latitude": round(lat, 6),
                    "Longitude": round(lon, 6),
                    "Nadir_Best_Image": "N/A",
                    "Nadir_Original_Image": r.get("Nadir_Original_Image", "N/A"),
                    "Nadir_Image_Precision_Cropped": False,
                    "Oblique_Best_Image": "N/A",
                    "Oblique_Original_Image": "N/A",
                    "Oblique_Image_Precision_Cropped": False,
                    "Needs_Review": r.get("Needs_Review", False),
                    "Possible_Vehicle": is_possible_vehicle
                })
                house_corners[new_uid] = r.get("corners_3d")
            else:
                if os.path.exists(old_ply_path): os.remove(old_ply_path)
                if os.path.exists(old_img_path): os.remove(old_img_path)

    print(f"      -> Kept {len(final_measurements)} / {len(raw_res)} candidates as valid buildings")
    if cfg["vehicle_rejection"]["reject_small_structures"]:
        print(f"      -> vehicle_rejection.reject_small_structures dropped {rejected_vehicle_count} candidate(s) "
              f"under {cfg['vehicle_rejection']['max_area_sqft']:.0f} sqft / "
              f"{cfg['vehicle_rejection']['max_height_ft']:.0f} ft entirely (not written to measurements.csv) - "
              "some may have been real small structures.")

    step_done("Step 9/10 (artifact purge + georeferencing)")
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

    # No longer gated on Original_Image == "N/A" - that also fired for every
    # merged (multi-building) candidate, which have a real corners_3d/
    # centroid and are perfectly precision-croppable, they just never got a
    # naive placeholder image at extraction time like normal candidates do.
    # find_best_registered_image already no-ops cheaply on its own (returns
    # None, None) when there's genuinely no image index or no corners_3d, so
    # there's nothing a skip here would protect.
    no_nadir_count, no_oblique_count = 0, 0
    cropped_oblique_count = 0
    for record in final_measurements:
        house_id = record["house_ID"]
        corners_3d = house_corners.get(house_id)

        nadir_src, nadir_ok = try_precision_crop(
            house_id, "nadir", corners_3d, nadir_img_data, reconstruction_data, out)
        if nadir_ok:
            record["Nadir_Original_Image"] = nadir_src
            record["Nadir_Image_Precision_Cropped"] = True
            cropped_count += 1
        else:
            # No registered candidate in the pool actually framed the house -
            # leave whatever plain-nearest photo extraction already copied to
            # best_images/{house_id}_nadir.jpg in place (uncropped) rather
            # than end up with nothing at all.
            no_nadir_count += 1

        oblique_src, oblique_ok = try_precision_crop(
            house_id, "oblique", corners_3d, oblique_img_data, reconstruction_data, out)
        if oblique_ok:
            record["Oblique_Original_Image"] = oblique_src
            record["Oblique_Image_Precision_Cropped"] = True
            cropped_oblique_count += 1
        else:
            # No naive fallback exists for oblique at all (see worker_extraction/
            # merge_overlapping_footprints) - failure here just means no
            # oblique photo for this house, not an uncropped placeholder.
            no_oblique_count += 1

        # Best_Image paths reflect whatever actually ended up on disk, not an
        # assumption from earlier - computed here because this is the only
        # point that has seen every possible source (extraction-time naive
        # placeholder, a merged candidate's carried-forward placeholder, or a
        # fresh precision crop just written above).
        nadir_path = os.path.join(out, "best_images", f"{house_id}_nadir.jpg")
        record["Nadir_Best_Image"] = nadir_path if os.path.exists(nadir_path) else "N/A"
        oblique_path = os.path.join(out, "best_images", f"{house_id}_oblique.jpg")
        record["Oblique_Best_Image"] = oblique_path if os.path.exists(oblique_path) else "N/A"

    print(f"      -> Nadir: cropped {cropped_count} / {len(final_measurements)} "
          f"({no_nadir_count} had no registered/in-frame candidate in the pool - left uncropped)")
    print(f"      -> Oblique: cropped {cropped_oblique_count} / {len(final_measurements)} "
          f"({no_oblique_count} had no registered/in-frame candidate in the pool - no oblique photo for these)")

    measurements_path = os.path.join(out, "measurements.csv")
    with open(measurements_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "house_ID", "Area_sqft", "Volume_cuft", "Height_ft", "Ratio_Area_to_Height",
            "X_UTM", "Y_UTM", "Latitude", "Longitude",
            "Nadir_Best_Image", "Nadir_Original_Image", "Nadir_Image_Precision_Cropped",
            "Oblique_Best_Image", "Oblique_Original_Image", "Oblique_Image_Precision_Cropped",
            "Needs_Review", "Possible_Vehicle"
        ])
        writer.writeheader()
        writer.writerows(final_measurements)
    print(f"      -> Measurements (incl. location) written to: {measurements_path}")
    print(f"      -> NOTE: floor is the MEDIAN ground-model elevation across each footprint "
          f"(see diagnostics/ground_surface.ply), not the minimum.")
    step_done("Step 10/10 (image cropping)")

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
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml"),
                        help="Path to a config.yml (ground model cell size/opening span/relevel, vehicle-"
                             "rejection thresholds, nadir/oblique pitch threshold, RANSAC point cap - see "
                             "config.yml itself for what each setting does). Defaults to config.yml next to "
                             "this script; missing keys fall back to the built-in defaults, and a missing "
                             "file falls back to defaults for everything. Pass a different path for a "
                             "per-project config instead of editing the default one in place.")

    args = parser.parse_args()
    process_reconstruction(args)
