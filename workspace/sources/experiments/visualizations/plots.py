import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ...local_datasets.dataset import Dataset
from sklearn.metrics import confusion_matrix
from ..metrics import Loss
from .utils import METRICS_PLOT_NAMES_MAPPING
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(artifacts_path, by_metric=Loss, dataset_name=None, output_dir='assets/', show_title=False):
    """
    Create and save confusion matrix plot for given metric.
    """
    # Font size configuration
    TITLE_FONT_SIZE = 18
    LABEL_FONT_SIZE = 16
    ANNOTATION_FONT_SIZE = 14
    CM_LABEL_FONT_SIZE = 12
    HEATBAR_FONT_SIZE = 14

    # Get artifacts
    by_metric_dir_path = os.path.join(artifacts_path, 'evaluation', f'by_{by_metric.name}')

    # Load predictions and labels
    predictions_path = os.path.join(by_metric_dir_path,
                                    'predictions.json')
    labels_path = os.path.join(by_metric_dir_path,
                               'labels.json')

    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    # Get label mapping and create confusion matrix
    label_mapping = Dataset.LABELS_MAPPING
    cm = confusion_matrix(labels, predictions)

    # Create plot
    plt.figure(figsize=(8, 6))

    # Calculate normalized confusion matrix
    total = cm.sum()
    cm_percentages = cm / total

    # Create heatmap
    hmap = sns.heatmap(cm_percentages, annot=True, fmt='.0%', cmap='Blues',
                       xticklabels=[label_mapping[0], label_mapping[1]],
                       yticklabels=[label_mapping[0], label_mapping[1]])

    # Set font size for tick labels
    plt.setp(hmap.get_xticklabels(), size=LABEL_FONT_SIZE)
    plt.setp(hmap.get_yticklabels(), size=LABEL_FONT_SIZE)

    # Increase colorbar ticks font size
    cbar = hmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=HEATBAR_FONT_SIZE)

    # Get the percentage labels
    texts = hmap.texts
    for text in texts:
        text.set_fontsize(ANNOTATION_FONT_SIZE)

    # Add count annotations and confusion matrix labels
    confusion_labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(len(cm)):
        for j in range(len(cm)):
            text_color = texts[i * len(cm) + j].get_color()

            # Add count annotation
            plt.text(j + 0.5, i + 0.65, f'({cm[i, j]})',
                     ha='center', va='center', color=text_color,
                     fontsize=ANNOTATION_FONT_SIZE)

            # Add confusion matrix label
            plt.text(j + 0.04, i + 0.07, confusion_labels[i][j],
                     ha='left', va='center', color=text_color,
                     fontsize=CM_LABEL_FONT_SIZE)

    metric_plot_name = METRICS_PLOT_NAMES_MAPPING[by_metric.name]
    if show_title:
        plt.title(f'Confusion matrix with best model selected by {metric_plot_name} metric',
                  fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=15)
    plt.xlabel('Predicted labels', fontweight='bold', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('True labels', fontweight='bold', fontsize=LABEL_FONT_SIZE)

    output_dir = os.path.join(output_dir, dataset_name) if dataset_name else output_dir
    output_dir = os.path.join(output_dir, 'confusion_matrix')
    os.makedirs(output_dir, exist_ok=True)
    # Save plot
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_by_{by_metric.name.lower()}.png'),
                bbox_inches='tight', dpi=300)
    plt.show()


plot_confusion_matrix(artifacts_path)


def plot_roc_curve(artifacts_path, by_metric=Loss, dataset_name=None, output_dir='assets/', show_title=False):
    """
    Plot ROC curve for a given metric from MLflow artifacts
    """
    # Font size configuration
    TITLE_FONT_SIZE = 18
    TICKS_LABELS_FONT_SIZE = 14
    LABEL_FONT_SIZE = 16
    LEGEND_FONT_SIZE = 14

    by_metric_dir_path = os.path.join(artifacts_path, 'evaluation', f'by_{by_metric.name}')

    # Load predictions and labels
    predictions_path = os.path.join(by_metric_dir_path,
                                    'predictions.json')
    labels_path = os.path.join(by_metric_dir_path,
                               'labels.json')

    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    y_true = np.array(labels)
    y_pred = np.array(predictions)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))

    # Create the base layer with fill
    plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange', zorder=1)

    # Create a separate axes for the lines and legend
    ax2 = plt.gca()
    ax2.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})', zorder=3)
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', zorder=3)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positives Rate', fontweight='bold', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('True Positives Rate', fontweight='bold', fontsize=LABEL_FONT_SIZE)
    metric_plot_name = METRICS_PLOT_NAMES_MAPPING[by_metric.name]
    if show_title:
        plt.title(f'ROC Curve of best model selected by {metric_plot_name} metric',
                  fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=15)
    legend = plt.legend(loc="lower right", framealpha=1, facecolor='white', fontsize=LEGEND_FONT_SIZE)
    plt.grid(True)

    # Set font size for tick labels
    plt.xticks(fontsize=TICKS_LABELS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_LABELS_FONT_SIZE)

    output_dir = os.path.join(output_dir, dataset_name) if dataset_name else output_dir
    output_dir = os.path.join(output_dir, 'roc_curve')
    os.makedirs(output_dir, exist_ok=True)
    # Save plot
    plt.savefig(os.path.join(output_dir, f'roc_curve_by_{by_metric.name.lower()}.png'),
                bbox_inches='tight', dpi=300)
    plt.show()
