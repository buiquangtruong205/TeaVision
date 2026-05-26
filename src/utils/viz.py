import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(matrix, class_names, output_path=None):
    ax = sns.heatmap(matrix, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    return ax
