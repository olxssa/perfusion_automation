import os
import argparse
import json
import numpy as np
import tifffile
from datetime import datetime
import matplotlib
from multi_well_perfusion import parse_well_timings, get_global_y_limits

# Set matplotlib backend to non-interactive before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cellpose import models
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Cellpose Worker Process")
    t0 = datetime.now()

    # Shared arguments
    parser.add_argument("--mode", required=True, choices=["segmentation", "analysis"], help="Execution mode")
    parser.add_argument("--bf_image", required=True, help="Path to Brightfield image")
    parser.add_argument("--fluo_image", required=True, help="Path to Fluorescent image (or stack)")
    parser.add_argument("--output", required=True, help="Path to output JSON results")
    parser.add_argument("--well", required=True, type=int, help="Current well number")
    parser.add_argument("--iteration", required=True, type=int, help="Current iteration/fov loop index")

    # Mode-specific argument
    parser.add_argument("--analysis_dir", required=False,
                        help="Directory to save overlay and trajectory plots (Required if mode is 'analysis')")

    # Optional dynamic Y-axis limit scaling parameters
    parser.add_argument("--ylim_min", type=float, default=None, help="Force minimum Y-axis limit")
    parser.add_argument("--ylim_max", type=float, default=None, help="Force maximum Y-axis limit")

    args = parser.parse_args()

    # Validate that analysis_dir is provided if analysis mode is active
    if args.mode == "analysis" and not args.analysis_dir:
        parser.error("--analysis_dir is required when --mode is 'analysis'")

    # Load model
    model = models.CellposeModel(gpu=True)

    # Read frames
    im_bf = tifffile.imread(args.bf_image)
    im_fluo_raw = tifffile.imread(args.fluo_image, key=slice(None))

    # Resolve fluorescence image structure depending on data shape/mode
    if args.mode == "segmentation":
        im_fluo_first_frame = im_fluo_raw
    else:  # analysis mode
        if im_fluo_raw.ndim == 2:
            im_fluo_raw = np.expand_dims(im_fluo_raw, axis=0)
        im_fluo_first_frame = im_fluo_raw[0]

    # Run Cellpose segmentation on Brightfield
    masks, flows, styles = model.eval(im_bf, diameter=100, channels=[0, 0])
    t1 = datetime.now()
    print("eval time", t1 - t0)

    # Total cells and properties on first frame
    properties = regionprops(masks, intensity_image=im_fluo_first_frame)
    total_cells = len(properties)

    # --- OPTIMIZED BACKGROUND SUBTRACTION (Downsample by 20x for median speed) ---
    bg_pixels = im_fluo_first_frame[masks == 0]
    bg_baseline = np.median(bg_pixels[::20]) if bg_pixels.size > 0 else 0

    SIGNAL_BUFFER = 0.1
    fluorescent_labels = []

    # Fast check for fluorescent labels
    for prop in properties:
        if (prop.mean_intensity - bg_baseline) > SIGNAL_BUFFER:
            fluorescent_labels.append(prop.label)

    fluorescent_cell_count = len(fluorescent_labels)

    # OPTIMIZED: Vectorized mask generation using np.isin instead of a nested loop
    positive_cells_mask = np.where(np.isin(masks, fluorescent_labels), masks, 0)

    # Prepare fields for JSON packaging across modes
    normalized_trajectories = {}
    filtered_fluorescent_count = fluorescent_cell_count

    # Define pathing for native Cellpose save files
    base_dir = os.path.dirname(args.bf_image) if args.mode == "segmentation" else args.analysis_dir
    cellpose_file_path = os.path.join(base_dir, f"well_{args.well}_fov_{args.iteration}_seg.npy")

    # Save standard cellpose format dictionary natively (.npy)
    cellpose_data = {
        "masks": masks,
        "flows": flows,
        "styles": styles,
        "chan_choose": [0, 0],
        "img": im_bf,
        "fluorescent_masks": positive_cells_mask
    }
    np.save(cellpose_file_path, cellpose_data)

    # =========================================================================
    # MODE 1: SEGMENTATION PIPELINE
    # =========================================================================
    if args.mode == "segmentation":
        # Save total cell masks
        masks_filename = os.path.join(base_dir, f"masks_well{args.well}_fov{args.iteration}.tif")
        tifffile.imwrite(masks_filename, masks.astype(np.uint16))

        # Save positive (fluorescent) cell masks only
        pos_masks_filename = os.path.join(base_dir, f"pos_masks_well{args.well}_fov{args.iteration}.tif")
        tifffile.imwrite(pos_masks_filename, positive_cells_mask.astype(np.uint16))

        # Save output JSON for Segmentation Mode
        fluorescent_cell_labels = [str(label) for label in fluorescent_labels]
        result = {
            "total_cells": total_cells,
            "fluorescent_cell_count": fluorescent_cell_count,
            "calculated_background": float(bg_baseline),
            "cellpose_results_file": cellpose_file_path,
            "fluorescent_cells_valid": fluorescent_cell_labels,
            "trajectories": {}
        }
        with open(args.output, "w") as f:
            json.dump(result, f)

    # =========================================================================
    # MODE 2: ANALYSIS PIPELINE
    # =========================================================================
    elif args.mode == "analysis":
        # --- GENERATING THE COLOR SEGMENTATION OVERLAYS ---
        THICKNESS = 5
        all_boundaries = find_boundaries(masks, mode='inner')
        fluo_boundaries = find_boundaries(positive_cells_mask, mode='inner')

        if THICKNESS > 1:
            selem = disk(THICKNESS - 1)
            all_boundaries = binary_dilation(all_boundaries, footprint=selem)
            fluo_boundaries = binary_dilation(fluo_boundaries, footprint=selem)

        # Mask overlays
        all_overlay = np.zeros((*masks.shape, 4))
        all_overlay[all_boundaries] = [0, 0, 0, 1.0]  # Black boundaries

        green_overlay = np.zeros((*masks.shape, 4))
        green_overlay[fluo_boundaries] = [0, 1, 0, 1.0]  # Green boundaries

        # Brightfield contrast enhancement
        p2_bf, p98_bf = np.percentile(im_bf, (0.01, 99.99))
        im_bf_contrast = np.clip(im_bf, p2_bf, p98_bf)
        im_bf_contrast = (im_bf_contrast - p2_bf) / (p98_bf - p2_bf + 1e-5)

        # Fluorescent contrast enhancement
        p2_fl, p98_fl = np.percentile(im_fluo_first_frame, (0.01, 99.99))
        im_fluo_contrast = np.clip(im_fluo_first_frame, p2_fl, p98_fl)
        im_fluo_contrast = (im_fluo_contrast - p2_fl) / (p98_fl - p2_fl + 1e-5)

        plt.style.use('dark_background')

        # --- SAVE BRIGHTFIELD OVERLAY (50% SMALLER: 4x4 instead of 8x8) ---
        fig_bf, ax_bf = plt.subplots(figsize=(4.5, 4.5), facecolor='#18181b')
        ax_bf.set_facecolor('#18181b')
        ax_bf.imshow(im_bf_contrast, cmap="gray")
        ax_bf.imshow(all_overlay)
        ax_bf.set_title(f"Brightfield Segmentation (Total Cells: {total_cells})",
                        fontsize=7, fontweight='bold', pad=8, color='#e4e4e7')
        ax_bf.axis("off")

        bf_overlay_out_path = os.path.join(args.analysis_dir, f"well_{args.well}_bf_overlay.png")
        plt.tight_layout()
        plt.savefig(bf_overlay_out_path, dpi=270, facecolor='#18181b')
        plt.close(fig_bf)

        # --- SAVE FLUORESCENT OVERLAY (50% SMALLER: 4x4 instead of 8x8) ---
        fig_fl, ax_fl = plt.subplots(figsize=(4.5, 4.5), facecolor='#18181b')
        ax_fl.set_facecolor('#18181b')
        ax_fl.imshow(im_fluo_contrast, cmap="bone")
        ax_fl.imshow(all_overlay)
        ax_fl.imshow(green_overlay)
        ax_fl.set_title(f"All t=0 (Count: {fluorescent_cell_count} / {total_cells})",
                        fontsize=7, fontweight='bold', pad=8, color='#e4e4e7')
        ax_fl.axis("off")

        overlay_out_path = os.path.join(args.analysis_dir, f"well_{args.well}_overlay.png")
        plt.tight_layout()
        plt.savefig(overlay_out_path, dpi=270, facecolor='#18181b')
        plt.close(fig_fl)

        # --- GENERATING THE TRAJECTORY CHARTS OVER TIME ---
        cell_pixel_indices = {prop.label: (prop.coords[:, 0], prop.coords[:, 1]) for prop in properties}
        all_labels = list(cell_pixel_indices.keys())
        fluorescent_labels_set = set(fluorescent_labels)

        cell_trajectories = {label: [] for label in all_labels}
        num_timepoints = im_fluo_raw.shape[0]
        mask_h, mask_w = masks.shape

        # Cache the background coordinates mask matrix once
        bg_mask = (masks == 0)

        # 1. Compute Raw CTCF Values across time
        for t in range(num_timepoints):
            raw_frame = im_fluo_raw[t]
            raw_h, raw_w = raw_frame.shape
            start_y = (raw_h - mask_h) // 2
            start_x = (raw_w - mask_w) // 2
            im_fluo_t = raw_frame[start_y:start_y + mask_h, start_x:start_x + mask_w]

            bg_pixels_t = im_fluo_t[bg_mask]
            bg_baseline_t = np.median(bg_pixels_t[::20]) if bg_pixels_t.size > 0 else 0

            for label in all_labels:
                y_idx, x_idx = cell_pixel_indices[label]
                cell_pixels = im_fluo_t[y_idx, x_idx]

                integrated_density = np.sum(cell_pixels)
                cell_area = cell_pixels.size

                ctcf = integrated_density - (cell_area * bg_baseline_t)
                cell_trajectories[label].append(ctcf)

        # 2. Apply dF/F0 Normalization based on the first few frames
        BASELINE_FRAMES = min(5, num_timepoints)
        valid_fluorescent_labels = []

        for label, ctcf_list in cell_trajectories.items():
            if len(ctcf_list) != num_timepoints:
                continue

            ctcf_array = np.array(ctcf_list)
            f0 = np.mean(ctcf_array[:BASELINE_FRAMES])

            if abs(f0) > 1e-5:
                df_f0 = (ctcf_array - f0) / f0
            else:
                df_f0 = ctcf_array - f0

            # Outlier filter
            if np.any(df_f0 < -0.9) or np.any(df_f0 > 100.0):
                print(f"Warning: Cell label {label} excluded due to unphysiological trajectory values.")
                continue

            normalized_trajectories[label] = df_f0.tolist()
            if label in fluorescent_labels:
                valid_fluorescent_labels.append(label)

        filtered_fluorescent_count = len(valid_fluorescent_labels)

        # =========================================================================
        # SAVE RUN ANALYSIS METADATA TO JSON FIRST
        # (This guarantees get_global_y_limits can safely include this well's data!)
        # =========================================================================
        json_safe_trajectories = {str(k): v for k, v in normalized_trajectories.items()}
        target_labels = valid_fluorescent_labels
        fluorescent_cell_labels = [str(label) for label in target_labels]

        result = {
            "total_cells": total_cells,
            "fluorescent_cell_count": filtered_fluorescent_count,
            "calculated_background": float(bg_baseline),
            "cellpose_results_file": cellpose_file_path,
            "fluorescent_cells_valid": fluorescent_cell_labels,
            "trajectories": json_safe_trajectories
        }
        with open(args.output, "w") as f:
            json.dump(result, f)

        # --- GET WELL TIMINGS FROM LOG ---
        log_file_path = os.path.join(os.path.dirname(args.analysis_dir),
                                     f"{os.path.basename(os.path.dirname(args.analysis_dir))}_log.csv")
        img_freq_s, perfusion_events = parse_well_timings(log_file_path, args.well)

        # Build accurate x-axis timepoints (in seconds)
        time_axis = [t * img_freq_s for t in range(num_timepoints)]

        # --- SAVE SINGLE WELL KINETIC TRAJECTORY PLOT ---
        fig_traj, ax_traj = plt.subplots(figsize=(8, 4.5), facecolor='#18181b')
        ax_traj.set_facecolor('#18181b')

        if normalized_trajectories:
            fluo_intensities = [
                normalized_trajectories[label] for label in valid_fluorescent_labels
                if label in normalized_trajectories
            ]

            # Trace individual trajectories using actual time steps
            for label in valid_fluorescent_labels:
                if label in normalized_trajectories:
                    intensities = normalized_trajectories[label]
                    ax_traj.plot(time_axis, intensities, alpha=0.15, linewidth=0.8, color="#71717a")

            if fluo_intensities:
                trajectories_matrix = np.array(fluo_intensities)
                population_mean = np.mean(trajectories_matrix, axis=0)

                # Signature mean trajectory plot
                ax_traj.plot(time_axis, population_mean, color='red', linewidth=2,
                             label=r'MEAN $\Delta$F/F$_0$', marker='o' if num_timepoints < 15 else None, markersize=3)

                # --- PLOT SOLUTION PERFUSION ANNOTATION LINES (DYNAMIC MATCHING) ---
                colors = [
                    "#3b82f6",  # Vibrant Blue
                    "#ec4899",  # Magenta/Pink
                    "#10b981",  # Emerald Green
                    "#f59e0b",  # Amber/Orange
                    "#a855f7",  # Purple
                    "#06b6d4",  # Cyan
                    "#f43f5e"   # Rose/Red
                ]
                solution_to_color_map = {}
                color_idx = 0

                # Built-in shortcut for this specific axis
                trans_single = ax_traj.get_xaxis_transform()

                for event in perfusion_events:
                    t_event = event["time"]
                    raw_name = event["solution"].strip()
                    norm_name = raw_name.lower()

                    # Assign color dynamically by name
                    if norm_name not in solution_to_color_map:
                        solution_to_color_map[norm_name] = colors[color_idx % len(colors)]
                        color_idx += 1

                    line_color = solution_to_color_map[norm_name]

                    # Vertical indicator line
                    ax_traj.axvline(x=t_event, color=line_color, linestyle="--", alpha=0.8,
                                    linewidth=1.2)

                    # Position text ending exactly at the top (Y=0.98 in axes fraction coordinate)
                    ax_traj.text(t_event + (img_freq_s * 0.2), 0.98, raw_name,
                                 rotation=90, transform=trans_single,
                                 horizontalalignment='left', verticalalignment='top',
                                 color=line_color, fontsize=7, fontweight='bold')

                ax_traj.legend(loc="upper right", frameon=True, facecolor='#09090b', edgecolor='#27272a', fontsize=7)
            else:
                ax_traj.text(0.5, 0.5, "No Fluorescent Cells Passed QC", ha='center', va='center', color='#dc2626',
                             fontsize=7)
        else:
            ax_traj.text(0.5, 0.5, "No Cells Detected in Mask", ha='center', va='center', color='#dc2626', fontsize=7)

        # =========================================================================
        # APPLY UNIFIED SCALE AND OVERRIDES AT THE ABSOLUTE END
        # =========================================================================
        # Query global scale from all saved well trajectories (including this one!)
        y_min, y_max = get_global_y_limits(args.analysis_dir)

        # Allow CLI parameters to override limits independently
        final_ymin = args.ylim_min if args.ylim_min is not None else y_min
        final_ymax = args.ylim_max if args.ylim_max is not None else y_max

        # Secure limits
        ax_traj.set_ylim(final_ymin, final_ymax)

        ax_traj.set_title(f"Well {args.well} Normalized Intensity Profiles of Physiological Cells (n = {filtered_fluorescent_count})",
                          fontsize=9, fontweight='bold', pad=12, color='white')
        ax_traj.set_xlabel("Time [seconds]", fontsize=8, color='white')
        ax_traj.set_ylabel(r"Fluorescence Intensity ($\Delta$F/F$_0$)", fontsize=8, color='white')
        ax_traj.tick_params(axis='both', which='major', labelsize=8)
        ax_traj.grid(True, linestyle="--", alpha=0.1, color='#27272a')

        trajectory_out_path = os.path.join(args.analysis_dir, f"well_{args.well}_trajectory.png")
        plt.tight_layout()
        plt.savefig(trajectory_out_path, dpi=270, facecolor='#18181b')
        plt.close(fig_traj)

        print("analysis time", datetime.now() - t1)

    print(
        f"Cellpose {args.mode.upper()} finished. Coordinates saved to {cellpose_file_path}. Meta and metrics stored in JSON."
    )