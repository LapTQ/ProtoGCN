import json
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from pathlib import Path
from natsort import natsorted


def get_latest_log_file(folder_path):
    """
    Get the latest .log.json file in a folder

    Args:
        folder_path: Path to the folder

    Returns:
        Path to the latest .log.json file or None if not found
    """
    folder = Path(folder_path)
    if not folder.exists():
        return None

    log_files = list(folder.glob("*.log.json"))
    if not log_files:
        return None

    # Sort by modification time, get the latest
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    return str(latest_log)


def discover_child_models(parent_folder):
    """
    Discover all child model folders and their latest log files in natural order

    Args:
        parent_folder: Path to the parent folder containing child model folders

    Returns:
        List of tuples: [(log_file_path, folder_name), ...]
    """
    parent_path = Path(parent_folder)
    if not parent_path.exists():
        print(f"Error: Parent folder does not exist: {parent_folder}")
        return []

    # Get all subdirectories
    child_folders = [d for d in parent_path.iterdir() if d.is_dir()]

    # Sort naturally
    child_folders = natsorted(child_folders, key=lambda x: x.name)

    child_models = []
    for child_folder in child_folders:
        log_file = get_latest_log_file(child_folder)
        if log_file:
            folder_name = child_folder.name
            child_models.append((log_file, folder_name))
            print(f"  Found child: {folder_name} -> {Path(log_file).name}")
        else:
            print(f"  Warning: No log file found in {child_folder.name}")

    return child_models


def parse_log_file(log_file_path, metric_name):
    """
    Parse a single log file and extract accuracy data

    Args:
        log_file_path: Path to the log file

    Returns:
        dict: Contains train and val data
    """
    data = {"train_epochs": [], "train_acc": [], "val_epochs": [], "val_acc": []}

    with open(log_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                log_entry = json.loads(line)

                # Skip environment info line
                if "env_info" in log_entry:
                    continue

                # Parse training data
                if log_entry.get("mode") == "train":
                    epoch = log_entry.get("epoch")
                    metric = log_entry.get(metric_name)

                    if epoch is not None and metric is not None:
                        data["train_epochs"].append(epoch)
                        data["train_acc"].append(metric)

                # Parse validation data
                elif log_entry.get("mode") == "val":
                    epoch = log_entry.get("epoch")
                    metric = log_entry.get(metric_name)

                    if epoch is not None and metric is not None:
                        data["val_epochs"].append(epoch)
                        data["val_acc"].append(metric)

            except json.JSONDecodeError:
                continue

    return data


def parse_config_with_children(config_path, config_name, metric_name):
    """
    Parse a config and all its child models

    Args:
        config_path: Path to config folder containing child models
        config_name: Name of the config

    Returns:
        dict: Combined data from all child models with shifted epochs
    """
    print(f"\nProcessing config: {config_name}")
    print(f"Path: {config_path}")

    # Discover child models
    child_models = discover_child_models(config_path)

    if not child_models:
        print(f"  No child models found for {config_name}")
        return None

    print(f"  Found {len(child_models)} child models")

    # Combined data with shifted epochs
    combined_data = {
        "train_epochs": [],
        "train_acc": [],
        "val_epochs": [],
        "val_acc": [],
        "child_info": [],  # Track which epochs belong to which child
        "child_names": [],  # Store child names for reference
    }

    train_epoch_offset = 0
    val_epoch_offset = 0

    for i, (log_file, child_name) in enumerate(child_models):
        child_data = parse_log_file(log_file, metric_name)

        # Track child model info
        child_info = {
            "name": child_name,
            "train_start": train_epoch_offset,
            "val_start": val_epoch_offset,
        }

        combined_data["child_names"].append(child_name)

        # Process training data
        if child_data["train_epochs"] and child_data["train_acc"]:
            shifted_train_epochs = [
                e + train_epoch_offset for e in child_data["train_epochs"]
            ]
            combined_data["train_epochs"].extend(shifted_train_epochs)
            combined_data["train_acc"].extend(child_data["train_acc"])

            # Update offset for next child
            max_epoch = max(child_data["train_epochs"])
            train_epoch_offset += max_epoch
            child_info["train_end"] = train_epoch_offset

            print(f"    {child_name}: {len(child_data['train_epochs'])} train points")

        # Process validation data
        if child_data["val_epochs"] and child_data["val_acc"]:
            shifted_val_epochs = [
                e + val_epoch_offset for e in child_data["val_epochs"]
            ]
            combined_data["val_epochs"].extend(shifted_val_epochs)
            combined_data["val_acc"].extend(child_data["val_acc"])

            # Update offset for next child
            max_epoch = max(child_data["val_epochs"])
            val_epoch_offset += max_epoch
            child_info["val_end"] = val_epoch_offset

            print(f"    {child_name}: {len(child_data['val_epochs'])} val points")

        combined_data["child_info"].append(child_info)

    return combined_data


def plot_multi_config_comparison(model_configs, output_folder, metric_name):
    """
    Plot training and validation curves for multiple configs (each with child models)

    Args:
        model_configs: List of tuples [(config_path, config_name), ...]
        output_folder: Folder path to save the figures
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Parse all configs with their children
    all_config_data = []
    config_names = []

    for config_path, config_name in model_configs:
        combined_data = parse_config_with_children(config_path, config_name, metric_name)
        if combined_data:
            all_config_data.append(combined_data)
            config_names.append(config_name)

    if not all_config_data:
        print("Error: No valid data found!")
        return None, None

    # Color palette for different configs
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if len(model_configs) > 10:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(model_configs)))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Calculate figure width based on maximum epoch span
    max_train_epochs = max(
        [
            max(data["train_epochs"]) if data["train_epochs"] else 0
            for data in all_config_data
        ]
    )
    max_val_epochs = max(
        [
            max(data["val_epochs"]) if data["val_epochs"] else 0
            for data in all_config_data
        ]
    )

    # Scale width: base width 10, add 0.05 per epoch, cap between 10 and 30
    train_width = min(max(10, 10 + max_train_epochs * 0.1), 30)
    val_width = min(max(10, 10 + max_val_epochs * 0.1), 30)

    # Plot 1: Training Accuracy with Shifted Epochs
    fig1, ax1 = plt.subplots(1, 1, figsize=(train_width, 8))

    for i, (data, config_name, color) in enumerate(
        zip(all_config_data, config_names, colors)
    ):
        if data["train_epochs"] and data["train_acc"]:
            marker = markers[i % len(markers)]

            # Plot the main line
            ax1.plot(
                data["train_epochs"],
                data["train_acc"],
                color=color,
                linewidth=2.5,
                label=f"{config_name} (Final: {data['train_acc'][-1]:.3f}, {len(data['child_info'])} models)",
                alpha=0.9,
            )

            # Add vertical lines and markers at child model boundaries
            for j, child_info in enumerate(data["child_info"]):
                if j > 0:  # Skip the first child (no boundary before it)
                    boundary_epoch = child_info.get("train_start", 0)
                    if boundary_epoch > 0:
                        # Add vertical dashed line
                        ax1.axvline(
                            x=boundary_epoch,
                            color=color,
                            linestyle="--",
                            linewidth=1.5,
                            alpha=0.5,
                        )

                        # Find the accuracy value at this boundary
                        try:
                            idx = data["train_epochs"].index(boundary_epoch)
                            boundary_acc = data["train_acc"][idx]
                            # Add a larger marker at the boundary
                            ax1.plot(
                                boundary_epoch,
                                boundary_acc,
                                marker="o",
                                markersize=10,
                                color=color,
                                markeredgecolor="white",
                                markeredgewidth=2,
                                zorder=5,
                            )
                        except ValueError:
                            pass

    ax1.set_xlabel("Continuous Epoch (Sequential Child Models)", fontsize=13)
    ax1.set_ylabel("Top-1 Accuracy", fontsize=13)
    ax1.set_title(
        "Training Accuracy - Multi-Config Comparison", fontsize=15, fontweight="bold"
    )
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    # ax1.set_ylim(0, 1)

    plt.tight_layout()
    train_save_path = os.path.join(output_folder, "training_accuracy_comparison.png")
    plt.savefig(train_save_path, dpi=300, bbox_inches="tight")
    print(f"\nTraining figure saved to: {train_save_path}")
    plt.close()

    # Plot 2: Validation Accuracy with Shifted Epochs
    fig2, ax2 = plt.subplots(1, 1, figsize=(val_width, 8))

    for i, (data, config_name, color) in enumerate(
        zip(all_config_data, config_names, colors)
    ):
        if data["val_epochs"] and data["val_acc"]:
            marker = markers[i % len(markers)]
            best_val_acc = max(data["val_acc"])
            best_val_idx = data["val_acc"].index(best_val_acc)
            best_val_epoch = data["val_epochs"][best_val_idx]

            # Plot the main line
            ax2.plot(
                data["val_epochs"],
                data["val_acc"],
                color=color,
                linewidth=2.5,
                label=f"{config_name} (Best: {best_val_acc:.3f}, {len(data['child_info'])} models)",
                alpha=0.9,
            )

            # Add star marker at best validation accuracy
            ax2.plot(
                best_val_epoch,
                best_val_acc,
                marker="*",
                markersize=20,
                color=color,
                markeredgecolor="gold",
                markeredgewidth=2,
                zorder=6,
            )

            # Add vertical lines and markers at child model boundaries
            for j, child_info in enumerate(data["child_info"]):
                if j > 0:  # Skip the first child (no boundary before it)
                    boundary_epoch = child_info.get("val_start", 0)
                    if boundary_epoch > 0:
                        # Add vertical dashed line
                        ax2.axvline(
                            x=boundary_epoch,
                            color=color,
                            linestyle="--",
                            linewidth=1.5,
                            alpha=0.5,
                        )

                        # Find the accuracy value at this boundary
                        try:
                            idx = data["val_epochs"].index(boundary_epoch)
                            boundary_acc = data["val_acc"][idx]
                            # Add a larger marker at the boundary
                            ax2.plot(
                                boundary_epoch,
                                boundary_acc,
                                marker="o",
                                markersize=10,
                                color=color,
                                markeredgecolor="white",
                                markeredgewidth=2,
                                zorder=5,
                            )
                        except ValueError:
                            pass

    ax2.set_xlabel("Continuous Epoch (Sequential Child Models)", fontsize=13)
    ax2.set_ylabel("Top-1 Accuracy", fontsize=13)
    ax2.set_title(
        "Validation Accuracy - Multi-Config Comparison", fontsize=15, fontweight="bold"
    )
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    # ax2.set_ylim(0, 1)

    plt.tight_layout()
    val_save_path = os.path.join(output_folder, "validation_accuracy_comparison.png")
    plt.savefig(val_save_path, dpi=300, bbox_inches="tight")
    print(f"Validation figure saved to: {val_save_path}")
    print(f"Figure dimensions - Train: {train_width}x8, Val: {val_width}x8")
    plt.close()

    # Print summary table
    print_comparison_summary(all_config_data, config_names)

    return train_save_path, val_save_path


def print_comparison_summary(all_config_data, config_names):
    """Print a comparison table of all configs"""

    print("\n" + "=" * 140)
    print("CONFIG COMPARISON SUMMARY")
    print("=" * 140)

    # Table header
    print(
        f"{'Config':<20} {'#Child':<8} {'Train Init':<12} {'Train Final':<12} "
        f"{'Val Init':<12} {'Val Final':<12} {'Val Best':<12} {'Best @ Child':<30} {'Best @ Epoch':<12}"
    )
    print("-" * 140)

    for data, config_name in zip(all_config_data, config_names):
        # Get values (handle empty lists)
        num_children = len(data["child_info"])
        train_init = data["train_acc"][0] if data["train_acc"] else 0
        train_final = data["train_acc"][-1] if data["train_acc"] else 0
        val_init = data["val_acc"][0] if data["val_acc"] else 0
        val_final = data["val_acc"][-1] if data["val_acc"] else 0
        val_best = max(data["val_acc"]) if data["val_acc"] else 0

        # Find which child model has the best validation accuracy
        best_child_name = "N/A"
        best_epoch_in_child = 0
        if data["val_acc"]:
            best_val_idx = data["val_acc"].index(val_best)
            best_val_epoch = data["val_epochs"][best_val_idx]

            # Find which child this epoch belongs to
            for i, child_info in enumerate(data["child_info"]):
                val_start = child_info.get("val_start", 0)
                val_end = child_info.get("val_end", 0)
                if val_start <= best_val_epoch < val_end:
                    best_child_name = (
                        data["child_names"][i]
                        if i < len(data["child_names"])
                        else child_info.get("name", "Unknown")
                    )
                    # Calculate epoch within the child model (not shifted)
                    best_epoch_in_child = best_val_epoch - val_start
                    break

        # Truncate long names
        display_child_name = (
            best_child_name[:28] + ".."
            if len(best_child_name) > 30
            else best_child_name
        )

        print(
            f"{config_name:<20} {num_children:<8} {train_init:<12.3f} {train_final:<12.3f} "
            f"{val_init:<12.3f} {val_final:<12.3f} {val_best:<12.3f} {display_child_name:<30} {best_epoch_in_child:<12}"
        )

    print("=" * 140)


def main():
    """
    Main function - configure your model configs here
    """

    # ============================================================================
    # CONFIGURE YOUR MODEL CONFIGS HERE: [(config_path, config_name), ...]
    # Each config_path should point to a folder containing child model folders
    # ============================================================================

    model_configs = [
        (
            "/home/laptq/laptq-fs26-shoplifting-detection/runs/ProtoGCN/fs26/v219--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2",
            "v219",
        ),
        (
            "/home/laptq/laptq-fs26-shoplifting-detection/runs/ProtoGCN/fs26/v220--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--b",
            "v220",
        ),
        (
            "/home/laptq/laptq-fs26-shoplifting-detection/runs/ProtoGCN/fs26/v221--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--jm",
            "v221",
        ),
        (
            "/home/laptq/laptq-fs26-shoplifting-detection/runs/ProtoGCN/fs26/v222--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--bm",
            "v222",
        )
    ]
    # metric_name = "top1_acc"
    metric_name = "harmonic_mean_recall"

    # Output folder to save the figures
    output_folder = "/home/laptq/laptq-fs26-shoplifting-detection/outputs/trivials/model_comparison"
    # output_folder = "/home/laptq/ProtoGCN/outputs/trivials/model_comparison"

    # ============================================================================

    print(f"Comparing {len(model_configs)} configs...")
    print(f"Figures will be saved to: {output_folder}")

    # Create comparison plots with auto-refresh
    while True:
        train_path, val_path = plot_multi_config_comparison(
            model_configs, output_folder, metric_name
        )
        if train_path and val_path:
            print(f"\nRefreshing in 30 seconds...\n")
        time.sleep(30)


if __name__ == "__main__":
    main()
