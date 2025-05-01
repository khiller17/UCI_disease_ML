import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, labels=[0, 1], class_names=["Benign", "Malignant"], title="Confusion Matrix"):
    """
    Plots a labeled confusion matrix using matplotlib and sklearn.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of label values. Defaults to [0, 1].
        class_names (list, optional): Display names for classes. Defaults to ["Benign", "Malignant"].
        title (str, optional): Title for the confusion matrix plot. Defaults to "Confusion Matrix".

    Returns:
        None: Displays the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(title)
    plt.show()

def pointbiserial_correlation(x, y):
    """
    Computes the point-biserial correlation between a continuous feature and a binary target.

    Args:
        x (array-like): Continuous feature values.
        y (array-like): Binary target values.

    Returns:
        float: Point-biserial correlation coefficient.
    """
    return stats.pointbiserialr(x, y).correlation