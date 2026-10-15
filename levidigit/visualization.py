"""Publication-oriented 3D and oblique-slice figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy import ndimage
from skimage.measure import marching_cubes

from .digit_length import DigitLengthResult


COLORS = {
    "tv": "#56B4E9",
    "bv": "#D55E00",
    "length": "#0047AB",
    "concavity": "#CC79A7",
    "centroid": "#FFFFFF",
    "endpoint": "#009E73",
}


def _voxel_to_world(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return points @ affine[:3, :3].T + affine[:3, 3]


def _display_frame(
    roi_mask: np.ndarray, affine: np.ndarray, result: DigitLengthResult
) -> tuple[np.ndarray, np.ndarray]:
    origin = result.concavity_centroid_world_mm
    longitudinal = result.distal_endpoint_world_mm - origin
    longitudinal /= np.linalg.norm(longitudinal)
    coords = np.argwhere(roi_mask)
    if len(coords) > 100_000:
        coords = coords[:: int(np.ceil(len(coords) / 100_000))]
    world = _voxel_to_world(coords, affine)
    covariance = np.cov((world - world.mean(axis=0)).T)
    _, eigenvectors = np.linalg.eigh(covariance)
    candidates = eigenvectors[:, ::-1].T
    side = max(candidates, key=lambda vector: 1.0 - abs(float(vector @ longitudinal)))
    side = side - longitudinal * float(side @ longitudinal)
    side /= np.linalg.norm(side)
    references = [affine[:3, index] for index in range(3)]
    reference = max(references, key=lambda vector: abs(float(vector @ side)))
    if float(reference @ side) < 0:
        side *= -1
    vertical = np.cross(longitudinal, side)
    vertical /= np.linalg.norm(vertical)
    # Display x=side, y=longitudinal, z=vertical. The measurement direction is
    # therefore identical across samples before applying the fixed camera.
    basis = np.column_stack((side, longitudinal, vertical))
    return origin, basis


def _native_xy_camera_frame(
    affine: np.ndarray, result: DigitLengthResult
) -> tuple[np.ndarray, np.ndarray]:
    """Align the digit in native XY, then permit a fixed 30-degree camera tilt."""
    origin = result.concavity_centroid_world_mm
    axis_i = affine[:3, 0] / np.linalg.norm(affine[:3, 0])
    axis_j = affine[:3, 1] / np.linalg.norm(affine[:3, 1])
    normal = np.cross(axis_i, axis_j)
    normal /= np.linalg.norm(normal)
    direction = result.distal_endpoint_world_mm - origin
    horizontal = direction - normal * float(direction @ normal)
    if np.linalg.norm(horizontal) < 1e-8:
        horizontal = axis_i.copy()
    horizontal /= np.linalg.norm(horizontal)
    if float(horizontal @ direction) < 0:
        horizontal *= -1
    vertical = np.cross(normal, horizontal)
    vertical /= np.linalg.norm(vertical)
    basis = np.column_stack((horizontal, vertical, normal))
    return origin, basis


def _transform(points: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (points - origin) @ basis


def _crop(mask: np.ndarray, padding: int = 4) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    coords = np.argwhere(mask)
    lower = np.maximum(coords.min(axis=0) - padding, 0)
    upper = np.minimum(coords.max(axis=0) + padding + 1, mask.shape)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    return slices, lower


def _mesh(
    ax,
    mask: np.ndarray,
    affine: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    color: str,
    alpha: float,
    *,
    max_dimension: int = 110,
    zorder: int = 1,
) -> None:
    if np.count_nonzero(mask) < 4:
        return
    slices, lower = _crop(mask)
    volume = mask[slices].astype(np.uint8)
    factor = max(1, int(np.ceil(max(volume.shape) / max_dimension)))
    reduced = volume[::factor, ::factor, ::factor]
    padded = np.pad(reduced, 1)
    vertices, faces, _, _ = marching_cubes(padded, level=0.5)
    vertices = (vertices - 1.0) * factor + lower
    world = _voxel_to_world(vertices, affine)
    shown = _transform(world, origin, basis)
    collection = Poly3DCollection(shown[faces], linewidths=0.05)
    collection.set_facecolor(color)
    collection.set_edgecolor(color)
    collection.set_alpha(alpha)
    collection.set_zorder(zorder)
    ax.add_collection3d(collection)


def _voxel_cloud(
    ax,
    mask: np.ndarray,
    affine: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    color: str,
    *,
    maximum_points: int = 6000,
    point_size: float = 3.0,
    alpha: float = 0.30,
    zorder: int = 2,
) -> None:
    voxels = np.argwhere(mask)
    if not len(voxels):
        return
    stride = max(1, int(np.ceil(len(voxels) / maximum_points)))
    shown = _transform(_voxel_to_world(voxels[::stride], affine), origin, basis)
    ax.scatter(
        shown[:, 0], shown[:, 1], shown[:, 2], s=point_size,
        color=color, alpha=alpha, depthshade=False, linewidths=0, zorder=zorder,
    )


def _equal_axes(ax, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.28)
    ax.set_axis_off()
    # Face-on native XY is elev=90. Elev=60 tilts 30 degrees away from it;
    # azim=-90 rotates about the displayed digit-length axis.
    ax.view_init(elev=60, azim=-90)


def _counted_line_runs(result: DigitLengthResult, affine: np.ndarray) -> list[np.ndarray]:
    voxels = np.argwhere(result.counted_line_mask)
    if not len(voxels):
        return []
    world = _voxel_to_world(voxels, affine)
    direction = result.distal_endpoint_world_mm - result.concavity_centroid_world_mm
    order = np.argsort((world - result.concavity_centroid_world_mm) @ direction)
    ordered = world[order]
    maximum_step = max(np.linalg.norm(affine[:3, axis]) for axis in range(3)) * 2.1
    breaks = np.where(np.linalg.norm(np.diff(ordered, axis=0), axis=1) > maximum_step)[0] + 1
    return [run for run in np.split(ordered, breaks) if len(run)]


def _save_pair(figure: plt.Figure, base_path: Path, dpi: int) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    # Reuse the rendered pixels instead of making Matplotlib draw an expensive
    # 3D scene twice. PNG and TIFF therefore contain the exact same figure.
    with Image.open(png_path) as rendered:
        rendered.convert("RGB").save(
            base_path.with_suffix(".tiff"),
            format="TIFF",
            compression="tiff_lzw",
            dpi=(dpi, dpi),
        )


def save_3d_figure(
    *,
    image: np.ndarray,
    roi_mask: np.ndarray,
    bv_mask: np.ndarray,
    affine: np.ndarray,
    result: DigitLengthResult,
    threshold: float,
    output_base: Path,
    dpi: int = 300,
) -> None:
    origin, basis = _native_xy_camera_frame(affine, result)
    roi_points = np.argwhere(roi_mask)
    shown_roi = _transform(
        _voxel_to_world(
            roi_points[:: max(1, len(roi_points) // 50_000)], affine
        ),
        origin,
        basis,
    )
    roi_slices, _ = _crop(roi_mask, padding=8)
    original_bone = np.zeros_like(roi_mask, dtype=bool)
    original_bone[roi_slices] = image[roi_slices] >= threshold

    figure = plt.figure(figsize=(15, 4.15))
    axes = [
        figure.add_subplot(1, 4, index, projection="3d", computed_zorder=False)
        for index in range(1, 5)
    ]
    titles = [
        f"A  Original bone (CT ≥ {threshold:g})",
        "B  TV segmentation",
        "C  Digit length audit",
        "D  BV/TV + digit length",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=0)
        _equal_axes(ax, shown_roi)

    _mesh(axes[0], original_bone, affine, origin, basis, "#6F7782", 0.58)
    _mesh(axes[1], roi_mask, affine, origin, basis, COLORS["tv"], 0.40)
    _mesh(axes[2], roi_mask, affine, origin, basis, COLORS["tv"], 0.035)
    _voxel_cloud(
        axes[2], result.concavity_mask, affine, origin, basis,
        COLORS["concavity"], alpha=0.24, zorder=3,
    )
    _mesh(axes[3], roi_mask, affine, origin, basis, COLORS["tv"], 0.035)
    _voxel_cloud(
        axes[3], bv_mask, affine, origin, basis, COLORS["bv"],
        maximum_points=18_000, point_size=2.0, alpha=0.13, zorder=2,
    )

    for line_world in _counted_line_runs(result, affine):
        line = _transform(line_world, origin, basis)
        for ax in (axes[2], axes[3]):
            ax.plot(
                line[:, 0], line[:, 1], line[:, 2], color=COLORS["length"],
                linewidth=4.0, zorder=10,
            )

    landmarks = _transform(
        np.vstack((result.concavity_centroid_world_mm, result.distal_endpoint_world_mm)),
        origin,
        basis,
    )
    axes[2].scatter(
        *landmarks[0], s=100, color=COLORS["centroid"], edgecolor="black",
        linewidth=0.5, depthshade=False, zorder=20,
    )
    axes[2].scatter(
        *landmarks[1], s=100, color=COLORS["endpoint"], edgecolor="black",
        linewidth=0.5, depthshade=False, zorder=20,
    )
    legend = [
        Patch(facecolor=COLORS["tv"], alpha=0.35, label="TV"),
        Patch(facecolor=COLORS["bv"], alpha=0.8, label="BV within TV"),
        Line2D([0], [0], color=COLORS["length"], lw=3, label="Digit length (counted)"),
        Patch(facecolor=COLORS["concavity"], alpha=0.7, label="Largest concavity"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["centroid"], markeredgecolor="black", label="Concavity centroid"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["endpoint"], markeredgecolor="black", label="Distal endpoint"),
    ]
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.14, top=0.88, wspace=-0.10)
    figure.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=6, fontsize=7.8, frameon=False,
    )
    _save_pair(figure, output_base, dpi)


def _sample_plane(
    volume: np.ndarray,
    affine: np.ndarray,
    center: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    horizontal_limits: tuple[float, float],
    vertical_limits: tuple[float, float],
    step_mm: float,
    order: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    width = min(800, max(2, int(np.ceil((horizontal_limits[1] - horizontal_limits[0]) / step_mm))))
    height = min(800, max(2, int(np.ceil((vertical_limits[1] - vertical_limits[0]) / step_mm))))
    xs = np.linspace(*horizontal_limits, width)
    ys = np.linspace(*vertical_limits, height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    world = center[:, None, None] + horizontal[:, None, None] * grid_x + vertical[:, None, None] * grid_y
    inverse = np.linalg.inv(affine)
    flat = world.reshape(3, -1)
    voxel = inverse[:3, :3] @ flat + inverse[:3, 3:4]
    sampled = ndimage.map_coordinates(volume, voxel, order=order, mode="constant", cval=0.0)
    return sampled.reshape(height, width), (xs[0], xs[-1], ys[0], ys[-1])


def _nice_scale_bar(width_mm: float) -> float:
    candidates = np.asarray([0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0])
    valid = candidates[candidates <= width_mm * 0.3]
    return float(valid[-1] if len(valid) else candidates[0])


def save_2d_figure(
    *,
    image: np.ndarray,
    roi_mask: np.ndarray,
    bv_mask: np.ndarray,
    affine: np.ndarray,
    result: DigitLengthResult,
    output_base: Path,
    audit: bool,
    dpi: int = 300,
) -> None:
    origin, basis = _display_frame(roi_mask, affine, result)
    side, longitudinal, vertical = basis.T
    roi_world = _voxel_to_world(np.argwhere(roi_mask), affine)
    relative = roi_world - origin
    projections = relative @ basis
    padding = max(0.04, float(np.max(np.ptp(projections, axis=0))) * 0.06)
    step = float(min(np.linalg.norm(affine[:3, axis]) for axis in range(3)))
    first_counted_world = _voxel_to_world(
        result.first_counted_voxel[None, :], affine
    )[0]
    midpoint = (first_counted_world + result.distal_endpoint_world_mm) / 2.0
    planes = [
        ("A  Longitudinal measurement plane", origin, side, longitudinal),
        ("B  Concavity-centered perpendicular", origin, side, vertical),
        ("C  Mid-length perpendicular", midpoint, side, vertical),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    overlays = [(bv_mask, COLORS["bv"], 0.42)]
    if audit:
        overlays.append((result.concavity_mask, COLORS["concavity"], 0.48))

    for ax, (title, center, horizontal, vertical_axis) in zip(axes, planes):
        centered = roi_world - center
        h_projection = centered @ horizontal
        v_projection = centered @ vertical_axis
        h_limits = (float(h_projection.min() - padding), float(h_projection.max() + padding))
        v_limits = (float(v_projection.min() - padding), float(v_projection.max() + padding))
        source, extent = _sample_plane(image, affine, center, horizontal, vertical_axis, h_limits, v_limits, step, 1)
        tv, _ = _sample_plane(roi_mask.astype(float), affine, center, horizontal, vertical_axis, h_limits, v_limits, step, 0)
        ax.imshow(source, cmap="gray", vmin=0, vmax=255, origin="lower", extent=extent, interpolation="nearest")
        if np.any(tv):
            ax.contour(tv, levels=[0.5], colors=[COLORS["tv"]], linewidths=1.4, origin="lower", extent=extent)
        for overlay_mask, color, alpha in overlays:
            sampled, _ = _sample_plane(overlay_mask.astype(float), affine, center, horizontal, vertical_axis, h_limits, v_limits, step, 0)
            masked = np.ma.masked_where(sampled < 0.5, sampled)
            ax.imshow(masked, cmap=ListedColormap([color]), alpha=alpha, origin="lower", extent=extent, interpolation="nearest")

        endpoints = np.vstack((result.concavity_centroid_world_mm, result.distal_endpoint_world_mm)) - center
        line_h = endpoints @ horizontal
        line_v = endpoints @ vertical_axis
        normal = np.cross(horizontal, vertical_axis)
        plane_distance = np.abs(endpoints @ normal)
        tolerance = step * 1.5
        for line_world in _counted_line_runs(result, affine):
            relative_line = line_world - center
            distances = np.abs(relative_line @ normal)
            visible = distances <= tolerance if "Longitudinal" not in title else np.ones(len(line_world), dtype=bool)
            if np.any(visible):
                ax.plot(
                    relative_line[visible] @ horizontal,
                    relative_line[visible] @ vertical_axis,
                    color=COLORS["length"], linewidth=2.2,
                )
        if audit:
            if plane_distance[0] <= tolerance:
                ax.scatter(line_h[0], line_v[0], s=34, color=COLORS["centroid"], edgecolor="black", linewidth=0.4, zorder=5)
            if plane_distance[1] <= tolerance:
                ax.scatter(line_h[1], line_v[1], s=34, color=COLORS["endpoint"], edgecolor="black", linewidth=0.4, zorder=5)
        bar = _nice_scale_bar(extent[1] - extent[0])
        x0 = extent[0] + 0.08 * (extent[1] - extent[0])
        y0 = extent[2] + 0.08 * (extent[3] - extent[2])
        ax.plot([x0, x0 + bar], [y0, y0], color="white", linewidth=3, solid_capstyle="butt")
        label = f"{bar * 1000:g} µm" if bar < 1 else f"{bar:g} mm"
        ax.text(x0 + bar / 2, y0 + 0.025 * (extent[3] - extent[2]), label, color="white", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_axis_off()

    legend = [
        Line2D([0], [0], color=COLORS["tv"], lw=2, label="TV boundary"),
        Patch(facecolor=COLORS["bv"], alpha=0.5, label="BV within TV"),
        Line2D([0], [0], color=COLORS["length"], lw=2, label="Digit length (counted)"),
    ]
    if audit:
        legend.append(Patch(facecolor=COLORS["concavity"], alpha=0.6, label="Largest concavity"))
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=len(legend), frameon=False, fontsize=9)
    _save_pair(figure, output_base, dpi)


def save_all_figures(
    *,
    case_id: str,
    image_nii: nib.spatialimages.SpatialImage,
    image: np.ndarray,
    roi_mask: np.ndarray,
    bv_mask: np.ndarray,
    result: DigitLengthResult,
    threshold: float,
    figures_dir: Path,
    dpi: int,
) -> None:
    affine = np.asarray(image_nii.affine)
    for audit, suffix in ((False, "standard"), (True, "audit")):
        save_3d_figure(
            image=image,
            roi_mask=roi_mask,
            bv_mask=bv_mask,
            affine=affine,
            result=result,
            threshold=threshold,
            output_base=figures_dir / "3d" / f"{case_id}_3d_{suffix}",
            audit=audit,
            dpi=dpi,
        )
        save_2d_figure(
            image=image,
            roi_mask=roi_mask,
            bv_mask=bv_mask,
            affine=affine,
            result=result,
            output_base=figures_dir / "2d" / f"{case_id}_2d_{suffix}",
            audit=audit,
            dpi=dpi,
        )
