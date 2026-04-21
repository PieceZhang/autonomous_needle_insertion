import numpy as np
import cv2

from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_erosion
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_dilation,
    binary_closing,
    disk
)
from skimage.measure import label, find_contours


# =========================================================
# Basic utilities
# =========================================================

def normalize_slice(slice_img):
    """Robustly normalize one slice to [0, 1]."""
    p1 = np.percentile(slice_img, 1)
    p99 = np.percentile(slice_img, 99)
    slice_clip = np.clip(slice_img, p1, p99)
    return (slice_clip - p1) / (p99 - p1 + 1e-8)


def keep_largest_component(mask):
    """Keep only the largest connected component."""
    lab = label(mask)
    if lab.max() == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest_label = np.argmax(counts)
    return lab == largest_label


def keep_components_touching_seed(candidate_mask, seed_mask):
    """
    Keep only connected components in candidate_mask
    that touch the seed_mask.
    """
    lab = label(candidate_mask)
    if lab.max() == 0:
        return np.zeros_like(candidate_mask, dtype=bool)

    touched_labels = np.unique(lab[(seed_mask) & (lab > 0)])
    if touched_labels.size == 0:
        return np.zeros_like(candidate_mask, dtype=bool)

    return np.isin(lab, touched_labels)


def build_filled_region(edge_mask,
                        seed_mask=None,
                        min_size=8,
                        close_radius=3,
                        dilation_radius=0,
                        hole_size=30):
    """
    Build a filled region from a candidate mask.
    """
    mask = remove_small_objects(edge_mask, min_size=min_size)

    if close_radius > 0:
        mask = binary_closing(mask, footprint=disk(close_radius))

    if seed_mask is not None:
        mask = keep_components_touching_seed(mask, seed_mask)

    mask = binary_fill_holes(mask)

    if seed_mask is not None:
        mask = keep_components_touching_seed(mask, seed_mask)

    mask = remove_small_objects(mask, min_size=min_size)
    mask = remove_small_holes(mask, area_threshold=hole_size)

    if dilation_radius > 0:
        mask = binary_dilation(mask, footprint=disk(dilation_radius))

    return mask


def get_outer_workspace_mask(slice_smooth,
                             outer_threshold=0.03,
                             min_workspace_size=500,
                             hole_size_workspace=200):
    """
    Extract workspace mask from the largest outer contour.
    """
    seed = slice_smooth > outer_threshold
    seed = remove_small_objects(seed, min_size=min_workspace_size)
    seed = remove_small_holes(seed, area_threshold=hole_size_workspace)
    seed = binary_closing(seed, footprint=disk(3))
    seed = keep_largest_component(seed)

    workspace_mask = binary_fill_holes(seed)
    return workspace_mask


def get_inner_workspace_mask(workspace_mask, inner_margin=12):
    """
    Remove a band near the workspace boundary.
    Only keep the inner region far enough from the outer contour.
    """
    if inner_margin <= 0:
        return workspace_mask.copy()

    inner_mask = binary_erosion(workspace_mask, structure=disk(inner_margin))
    return inner_mask


# =========================================================
# Dark obstacle rule:
# exclude dark band near outer boundary
# =========================================================

def segment_dark_obstacles_inner_only(slice_smooth,
                                      workspace_mask,
                                      dark_threshold=0.18,
                                      inner_margin=12,
                                      min_dark_size=6,
                                      close_radius=2,
                                      hole_size_dark=20,
                                      dilation_radius=1):
    """
    Dark obstacles are only counted in the inner workspace region,
    excluding the dark band near the outer boundary.
    """
    inner_workspace_mask = get_inner_workspace_mask(
        workspace_mask,
        inner_margin=inner_margin
    )

    dark_mask = inner_workspace_mask & (slice_smooth < dark_threshold)

    dark_mask = remove_small_objects(dark_mask, min_size=min_dark_size)

    if close_radius > 0:
        dark_mask = binary_closing(dark_mask, footprint=disk(close_radius))

    dark_mask = remove_small_holes(dark_mask, area_threshold=hole_size_dark)

    if dilation_radius > 0:
        dark_mask = binary_dilation(dark_mask, footprint=disk(dilation_radius))

    return dark_mask


# =========================================================
# Full segmentation
# =========================================================

def segment_workspace_and_obstacles(
    slice_img,
    outer_threshold=0.03,
    tissue_threshold=0.08,

    dark_threshold=0.18,
    dark_inner_margin=12,

    bright_seed_percentile=99.5,
    bright_edge_percentile=95.0,
    bright_inner_percentile=85.0,

    min_workspace_size=500,
    min_dark_size=6,
    min_bright_size=10,

    hole_size_workspace=200,
    hole_size_dark=20,
    hole_size_bright=30,

    smooth_sigma=0.8,

    dark_close_radius=2,
    dark_dilation_radius=1,

    bright_close_radius=4,
    bright_dilation_radius=1,

    bright_local_dilation_radius=8,
    bright_secondary_close_radius=5
):
    """
    Segment workspace and obstacles from one slice.
    """
    slice_norm = normalize_slice(slice_img)
    slice_smooth = gaussian_filter(slice_norm, sigma=smooth_sigma)

    workspace_mask = get_outer_workspace_mask(
        slice_smooth,
        outer_threshold=outer_threshold,
        min_workspace_size=min_workspace_size,
        hole_size_workspace=hole_size_workspace
    )

    tissue_mask = workspace_mask & (slice_smooth > tissue_threshold)
    tissue_mask = remove_small_objects(
        tissue_mask,
        min_size=max(20, min_workspace_size // 2)
    )

    dark_region_mask = segment_dark_obstacles_inner_only(
        slice_smooth=slice_smooth,
        workspace_mask=workspace_mask,
        dark_threshold=dark_threshold,
        inner_margin=dark_inner_margin,
        min_dark_size=min_dark_size,
        close_radius=dark_close_radius,
        hole_size_dark=hole_size_dark,
        dilation_radius=dark_dilation_radius
    )

    tissue_values = slice_smooth[tissue_mask]

    if tissue_values.size == 0:
        bright_region_mask = np.zeros_like(slice_smooth, dtype=bool)

    else:
        bright_seed_thr = np.percentile(tissue_values, bright_seed_percentile)
        bright_edge_thr = np.percentile(tissue_values, bright_edge_percentile)
        bright_inner_thr = np.percentile(tissue_values, bright_inner_percentile)

        bright_seed_mask = tissue_mask & (slice_smooth >= bright_seed_thr)
        bright_edge_mask = tissue_mask & (slice_smooth >= bright_edge_thr)

        bright_region_mask = build_filled_region(
            edge_mask=bright_edge_mask,
            seed_mask=bright_seed_mask,
            min_size=min_bright_size,
            close_radius=bright_close_radius,
            dilation_radius=bright_dilation_radius,
            hole_size=hole_size_bright
        )

        local_roi = binary_dilation(
            bright_region_mask,
            footprint=disk(bright_local_dilation_radius)
        )

        bright_inner_candidate = (
            local_roi &
            tissue_mask &
            (slice_smooth >= bright_inner_thr)
        )

        bright_inner_candidate = remove_small_objects(
            bright_inner_candidate,
            min_size=max(20, min_bright_size)
        )

        bright_inner_candidate = keep_components_touching_seed(
            bright_inner_candidate,
            bright_region_mask
        )

        bright_region_mask = bright_region_mask | bright_inner_candidate
        bright_region_mask = binary_closing(
            bright_region_mask,
            footprint=disk(bright_secondary_close_radius)
        )
        bright_region_mask = binary_fill_holes(bright_region_mask)
        bright_region_mask = bright_region_mask & tissue_mask
        bright_region_mask = keep_components_touching_seed(
            bright_region_mask,
            bright_seed_mask
        )
        bright_region_mask = remove_small_objects(
            bright_region_mask,
            min_size=min_bright_size
        )
        bright_region_mask = remove_small_holes(
            bright_region_mask,
            area_threshold=hole_size_bright
        )

    total_obstacle_mask = dark_region_mask | bright_region_mask

    return {
        "slice_norm": slice_norm,
        "workspace_mask": workspace_mask,
        "dark_region_mask": dark_region_mask,
        "bright_region_mask": bright_region_mask,
        "total_obstacle_mask": total_obstacle_mask
    }


# =========================================================
# Geometry / planning
# =========================================================

def trace_ray_ordered_from_point(point, direction, image_shape, max_len=None):
    """
    Trace an ordered integer ray from a point along a direction.
    Return points as (row, col), preserving order.
    """
    h, w = image_shape
    p = np.array(point, dtype=float)
    d = np.array(direction, dtype=float)

    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return np.empty((0, 2), dtype=int)

    d = d / norm

    if max_len is None:
        max_len = int(2 * np.hypot(h, w))

    pts = []
    seen = set()

    for alpha in range(max_len + 1):
        q = p + alpha * d
        r = int(round(q[0]))
        c = int(round(q[1]))

        if not (0 <= r < h and 0 <= c < w):
            break

        key = (r, c)
        if key not in seen:
            seen.add(key)
            pts.append(key)

    if len(pts) == 0:
        return np.empty((0, 2), dtype=int)

    return np.array(pts, dtype=int)


def choose_inside_start_point(ray_inside, inward_offset=10):
    """
    Choose a start point that lies inside workspace, near the exit side.
    ray_inside goes from target to boundary-inside endpoint.
    inward_offset means how many sampled points to step back from the boundary.
    """
    if len(ray_inside) == 0:
        raise ValueError("ray_inside is empty.")

    idx = max(0, len(ray_inside) - 1 - inward_offset)
    return tuple(ray_inside[idx])


def find_best_exit_start_from_target_by_angle(target,
                                              workspace_mask,
                                              obstacle_check,
                                              outside_offset=8,
                                              n_angles=720,
                                              inward_start_offset=10):
    """
    Search all outward directions from target.
    Keep all valid rays and choose the shortest one inside workspace.

    Returns:
        start_inside, start_outside, best_ray_inside
    """
    if not workspace_mask[target]:
        raise ValueError("Target must lie inside workspace.")

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    candidates = []

    for theta in angles:
        direction = np.array([np.sin(theta), np.cos(theta)], dtype=float)

        ray = trace_ray_ordered_from_point(
            point=target,
            direction=direction,
            image_shape=workspace_mask.shape
        )

        if len(ray) == 0:
            continue

        inside_flags = workspace_mask[ray[:, 0], ray[:, 1]]

        if not inside_flags[0]:
            continue

        inside_pts = []
        exited = False

        for (r, c), inside in zip(ray, inside_flags):
            if inside:
                inside_pts.append((r, c))
            else:
                exited = True
                break

        if len(inside_pts) < 2:
            continue

        if not exited:
            continue

        inside_pts = np.array(inside_pts, dtype=int)
        rr = inside_pts[:, 0]
        cc = inside_pts[:, 1]

        if np.any(obstacle_check[rr, cc]):
            continue

        last_inside = np.array(inside_pts[-1], dtype=float)
        d = direction / (np.linalg.norm(direction) + 1e-8)

        start_outside = None
        for extra in range(outside_offset, outside_offset + 40):
            candidate = last_inside + extra * d
            sr = int(round(candidate[0]))
            sc = int(round(candidate[1]))

            h, w = workspace_mask.shape
            if 0 <= sr < h and 0 <= sc < w:
                if not workspace_mask[sr, sc]:
                    start_outside = (sr, sc)
                    break
            else:
                sr = np.clip(sr, 0, h - 1)
                sc = np.clip(sc, 0, w - 1)
                start_outside = (sr, sc)
                break

        if start_outside is None:
            continue

        start_inside = choose_inside_start_point(
            inside_pts,
            inward_offset=inward_start_offset
        )

        path_len = len(inside_pts)

        candidates.append({
            "start_inside": start_inside,
            "start_outside": start_outside,
            "ray_inside": inside_pts,
            "path_len": path_len
        })

    if len(candidates) == 0:
        raise RuntimeError(
            "Failed to find a valid outward ray from target. "
            "Try another target or relax obstacle margin."
        )

    best = min(candidates, key=lambda x: x["path_len"])
    return best["start_inside"], best["start_outside"], best["ray_inside"]


def choose_default_target(workspace_mask, obstacle_mask=None):
    """
    Pick a valid target near the centroid of the available workspace.
    """
    valid = np.asarray(workspace_mask, dtype=bool).copy()
    if obstacle_mask is not None:
        valid &= ~np.asarray(obstacle_mask, dtype=bool)

    coords = np.argwhere(valid)
    if coords.size == 0:
        raise RuntimeError("No valid target point is available inside the workspace.")

    centroid = coords.mean(axis=0)
    best_idx = int(np.argmin(np.sum((coords - centroid[None, :]) ** 2, axis=1)))
    row, col = coords[best_idx]
    return int(row), int(col)


def plan_path_from_masks(workspace_mask,
                         obstacle_mask,
                         target,
                         margin=3,
                         outside_offset=8,
                         n_angles=360,
                         inward_start_offset=20):
    """
    Plan a straight line on precomputed masks.
    """
    target = tuple(int(v) for v in target)
    h, w = workspace_mask.shape
    row = int(np.clip(target[0], 0, h - 1))
    col = int(np.clip(target[1], 0, w - 1))
    target = (row, col)

    if not workspace_mask[target]:
        raise ValueError("Target must lie inside workspace.")

    if obstacle_mask[target]:
        raise ValueError("Target must not lie inside an obstacle.")

    obstacle_check = binary_dilation(
        obstacle_mask.astype(bool),
        footprint=disk(int(max(0, margin)))
    )

    start_inside, start_outside, ray_inside = find_best_exit_start_from_target_by_angle(
        target=target,
        workspace_mask=workspace_mask.astype(bool),
        obstacle_check=obstacle_check,
        outside_offset=outside_offset,
        n_angles=n_angles,
        inward_start_offset=inward_start_offset,
    )

    return {
        "target": target,
        "start_inside": start_inside,
        "start_outside": start_outside,
        "ray_inside": ray_inside,
    }


def _gray_to_rgb_uint8(gray_img):
    gray_img = np.asarray(gray_img)
    if gray_img.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")
    if gray_img.dtype != np.uint8:
        gray_img = normalize_slice(gray_img)
        gray_img = np.clip(gray_img * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.repeat(gray_img[:, :, None], 3, axis=2)


def render_planning_result_image(background_img,
                                 dark_region_mask,
                                 bright_region_mask,
                                 workspace_mask=None,
                                 target=None,
                                 start_inside=None,
                                 start_outside=None,
                                 ray_inside=None):
    """
    Render a planning overlay as an RGB uint8 image.
    """
    rgb = _gray_to_rgb_uint8(background_img).copy()
    rgb[dark_region_mask] = np.array([0, 255, 255], dtype=np.uint8)
    rgb[bright_region_mask] = np.array([255, 0, 0], dtype=np.uint8)

    if workspace_mask is not None:
        contours = find_contours(np.asarray(workspace_mask, dtype=float), level=0.5)
        if contours:
            outer = max(contours, key=lambda x: len(x))
            contour_pts = np.round(outer[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(rgb, [contour_pts], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    if ray_inside is not None and len(ray_inside) > 1:
        ray_pts = np.round(np.asarray(ray_inside)[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(rgb, [ray_pts], isClosed=False, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    if target is not None and start_inside is not None:
        cv2.line(
            rgb,
            (int(round(target[1])), int(round(target[0]))),
            (int(round(start_inside[1])), int(round(start_inside[0]))),
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    if start_inside is not None:
        center = (int(round(start_inside[1])), int(round(start_inside[0])))
        cv2.circle(rgb, center, 6, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    if start_outside is not None:
        center = (int(round(start_outside[1])), int(round(start_outside[0])))
        cv2.circle(rgb, center, 4, color=(0, 165, 255), thickness=2, lineType=cv2.LINE_AA)

    if target is not None:
        center = (int(round(target[1])), int(round(target[0])))
        cv2.drawMarker(
            rgb,
            center,
            color=(255, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=14,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    return rgb


# =========================================================
# Visualization and interaction
# =========================================================

def plot_only_outer_contour(ax, mask, color="yellow", linewidth=2):
    """
    Plot the longest outer contour.
    """
    contours = find_contours(mask.astype(float), level=0.5)
    if len(contours) == 0:
        return
    outer = max(contours, key=lambda x: len(x))
    ax.plot(outer[:, 1], outer[:, 0], color=color, linewidth=linewidth)


def choose_target_interactively(background_img,
                                workspace_mask,
                                dark_region_mask,
                                bright_region_mask):
    """
    left: raw image
    right: overlay
    """
    plt = _require_pyplot()
    overlay = np.dstack([background_img, background_img, background_img]).copy()
    overlay[dark_region_mask] = [0.0, 1.0, 1.0]   # cyan
    overlay[bright_region_mask] = [1.0, 0.0, 0.0] # red

    fig = plt.figure(figsize=(16, 8))

    ax_raw = fig.add_axes([0.06, 0.08, 0.40, 0.84])
    ax_overlay = fig.add_axes([0.46, 0.08, 0.40, 0.84])

    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
    except Exception:
        try:
            manager.resize(*manager.window.maxsize())
        except Exception:
            pass

    ax_raw.imshow(background_img, cmap="gray", aspect="equal")
    ax_raw.set_title("Original Slice", fontsize=16)
    ax_raw.axis("off")

    ax_overlay.imshow(overlay, aspect="equal")
    ax_overlay.set_title("Segmentation Overlay - Click one target point", fontsize=16)
    ax_overlay.axis("off")

    plt.sca(ax_overlay)
    clicked = plt.ginput(1, timeout=-1)

    if len(clicked) == 0:
        plt.close(fig)
        raise RuntimeError("No target point was selected.")

    x, y = clicked[0]
    col = int(round(x))
    row = int(round(y))

    h, w = workspace_mask.shape
    row = np.clip(row, 0, h - 1)
    col = np.clip(col, 0, w - 1)

    for ax in [ax_raw, ax_overlay]:
        ax.scatter(
            col, row,
            s=110,
            marker="x",
            c="magenta",
            linewidths=2.5,
            label="Target"
        )

    ax_overlay.set_title("Target selected, searching best outward direction...", fontsize=16)
    fig.canvas.draw()
    plt.pause(0.01)

    return fig, ax_raw, ax_overlay, (row, col)


def draw_result_on_existing_axes(ax_raw,
                                 ax_overlay,
                                 background_img,
                                 workspace_mask,
                                 dark_region_mask,
                                 bright_region_mask,
                                 start_inside,
                                 start_outside,
                                 target,
                                 ray_inside):
    """
    Plot the planning result on the existing axes.
    left: original slice with target/start/ray
    right: segmentation overlay with contour + target/start/ray
    """
    plt = _require_pyplot()
    overlay = np.dstack([background_img, background_img, background_img]).copy()
    overlay[dark_region_mask] = [0.0, 1.0, 1.0]   # cyan
    overlay[bright_region_mask] = [1.0, 0.0, 0.0] # red

    ax_raw.clear()
    ax_overlay.clear()

    ax_raw.imshow(background_img, cmap="gray")
    ax_raw.set_title("Original Slice")
    ax_raw.axis("off")

    ax_raw.plot(
        [target[1], start_inside[1]],
        [target[0], start_inside[0]],
        color="lime",
        linewidth=3
    )

    if len(ray_inside) > 1:
        ax_raw.plot(
            ray_inside[:, 1],
            ray_inside[:, 0],
            linestyle="--",
            color="white",
            linewidth=4
        )

    ax_raw.scatter(
        start_inside[1], start_inside[0],
        s=90,
        marker="o",
        facecolors="none",
        edgecolors="blue",
        linewidths=3
    )

    ax_raw.scatter(
        target[1], target[0],
        s=110,
        marker="x",
        c="magenta",
        linewidths=3
    )

    ax_overlay.imshow(overlay)
    ax_overlay.set_aspect('equal', adjustable='box')
    # plot_only_outer_contour(ax_overlay, workspace_mask, color="yellow", linewidth=2)

    ax_overlay.plot(
        [target[1], start_inside[1]],
        [target[0], start_inside[0]],
        color="lime",
        linewidth=3,
        label="Target-to-start line"
    )

    if len(ray_inside) > 1:
        ax_overlay.plot(
            ray_inside[:, 1],
            ray_inside[:, 0],
            linestyle="--",
            color="white",
            linewidth=4,
            label="Obstacle-free ray inside workspace"
        )

    ax_overlay.scatter(
        start_inside[1], start_inside[0],
        s=90,
        marker="o",
        facecolors="none",
        edgecolors="blue",
        linewidths=3,
        label="Start (inside workspace)"
    )

    ax_overlay.scatter(
        target[1], target[0],
        s=110,
        marker="x",
        c="magenta",
        linewidths=3,
        label="Target"
    )

    ax_overlay.set_title("Best straight-line planning (start shown inside workspace)")
    ax_overlay.axis("off")

    handles, labels = ax_overlay.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax_overlay.legend(
        uniq.values(),
        uniq.keys(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=13,
        frameon=True
    )

    ax_overlay.figure.canvas.draw()
    plt.pause(0.01)


def _require_pyplot():
    import matplotlib.pyplot as plt
    return plt


def main():
    import nibabel as nib

    plt = _require_pyplot()
    nii_path = "phantom_imaging_data/abdominal.nii.gz"
    k = 230     # choose which slice to show, suggest view0: 90-290, view1:180-300

    img = nib.load(nii_path)
    data = img.get_fdata()

    print("Volume shape:", data.shape)
    print("Voxel spacing:", img.header.get_zooms())

    slice_img = data[:, k, :]     # can also be data[k,:,:] or data[:,:,k] 

    results = segment_workspace_and_obstacles(
        slice_img,
        outer_threshold=0.03,
        tissue_threshold=0.08,

        dark_threshold=0.18,
        dark_inner_margin=13,  # larger margin for dark band exclusion, can be tuned based on image quality

        bright_seed_percentile=99.6,  # larger -> fewer bright seeds, smaller bright region size
        bright_edge_percentile=95.0,  # larger -> more strict bright edge, smaller bright region size
        bright_inner_percentile=87.0, # larger -> smaller bright region size

        min_workspace_size=500,
        min_dark_size=6,
        min_bright_size=10,

        hole_size_workspace=200,
        hole_size_dark=20,
        hole_size_bright=30,

        smooth_sigma=0.8,

        dark_close_radius=2,
        dark_dilation_radius=1,

        bright_close_radius=4,
        bright_dilation_radius=1,

        bright_local_dilation_radius=8,
        bright_secondary_close_radius=5
    )

    slice_norm = results["slice_norm"]
    workspace_mask = results["workspace_mask"]
    dark_region_mask = results["dark_region_mask"]
    bright_region_mask = results["bright_region_mask"]
    total_obstacle_mask = results["total_obstacle_mask"]

    fig, ax_raw, ax_overlay, target = choose_target_interactively(
        background_img=slice_norm,
        workspace_mask=workspace_mask,
        dark_region_mask=dark_region_mask,
        bright_region_mask=bright_region_mask
    )

    print("Selected target (row, col):", target)

    if not workspace_mask[target]:
        ax_overlay.set_title("Selected target is outside workspace")
        fig.canvas.draw()
        plt.pause(0.01)
        plt.show()
        raise ValueError("[ERROR] Please choose a point inside workspace.")

    if total_obstacle_mask[target]:
        ax_overlay.set_title("Selected target lies inside obstacle")
        fig.canvas.draw()
        plt.pause(0.01)
        plt.show()
        raise ValueError("[ERROR] Please choose a target outside obstacles.")

    margin = 3
    obstacle_check = binary_dilation(total_obstacle_mask, footprint=disk(margin))

    try:
        start_inside, start_outside, ray_inside = find_best_exit_start_from_target_by_angle(
            target=target,
            workspace_mask=workspace_mask,
            obstacle_check=obstacle_check,
            outside_offset=8,
            n_angles=360,       # change sample numbers
            inward_start_offset=20     # change steps back from boundary for inside start point
        ) 

        print("[INFO] Found inside start (row, col):", start_inside)
        print("[INFO] Found outside exit helper point (row, col):", start_outside)

        draw_result_on_existing_axes(
            ax_raw=ax_raw,
            ax_overlay=ax_overlay,
            background_img=slice_norm,
            workspace_mask=workspace_mask,
            dark_region_mask=dark_region_mask,
            bright_region_mask=bright_region_mask,
            start_inside=start_inside,
            start_outside=start_outside,
            target=target,
            ray_inside=ray_inside
        )

        plt.show()

    except RuntimeError as e:
        print("[ERROR] " + str(e))
        ax_overlay.set_title("[ERROR] No valid outward direction found for this target")
        fig.canvas.draw()
        plt.pause(0.01)
        plt.show()


if __name__ == "__main__":
    main()
