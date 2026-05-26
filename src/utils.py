import os
import matplotlib.pyplot as plt
import seaborn as sns

BRAND_PALETTE = {
    "deep_green": "#2E7D32",
    "light_sage": "#A5D6A7",
    "neutral_gray": "#37474F",
    "bg_gray": "#ECEFF1",
}


def log_folivus_banner():
    banner = f"""
    =======================================================
               🌿 Welcome to Project Folivus 🌿
       Production-Grade Crop Disease Diagnostics Engine
    =======================================================
    """
    print(banner)


def save_metrics_plot(history, output_path="reports/metrics_plot.png"):
    """
    Generates optimization summary performance curves tracking
    loss curves and categorical accuracy metrics.
    """
    sns.set_theme(style="whitegrid")
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), facecolor=BRAND_PALETTE["bg_gray"]
    )

    # Loss Optimization Progression
    ax1.plot(
        epochs,
        history["train_loss"],
        color=BRAND_PALETTE["neutral_gray"],
        linestyle="--",
        marker="o",
        label="Train Loss",
    )
    ax1.plot(
        epochs,
        history["val_loss"],
        color=BRAND_PALETTE["deep_green"],
        linewidth=2.5,
        marker="s",
        label="Val Loss",
    )
    ax1.set_title(
        "Folivus Cross-Entropy Loss Curve",
        fontsize=12,
        fontweight="bold",
        color=BRAND_PALETTE["neutral_gray"],
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss Metric Value")
    ax1.legend()

    # Categorical Accuracy Metrics
    ax2.plot(
        epochs,
        history["train_acc"],
        color=BRAND_PALETTE["neutral_gray"],
        linestyle="--",
        marker="o",
        label="Train Acc",
    )
    ax2.plot(
        epochs,
        history["val_acc"],
        color=BRAND_PALETTE["light_sage"],
        linewidth=2.5,
        marker="s",
        label="Val Acc",
    )
    ax2.set_title(
        "Folivus Network Classification Accuracy",
        fontsize=12,
        fontweight="bold",
        color=BRAND_PALETTE["neutral_gray"],
    )
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy Metric (0.0 - 1.0)")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
