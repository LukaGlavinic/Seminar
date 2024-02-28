import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def show_confusion_matrix(extracted_features_Y, predictions, indices_of_poisoned, save_path=None):

    gt_pois = [0] * len(predictions)
    for ind in indices_of_poisoned:
        gt_pois[ind] = 1

    pred_pois = [0] * len(predictions)
    for i, pred in enumerate(predictions):
        if pred != extracted_features_Y[i]:
            pred_pois[i] = 1
    
    confusion_mat = confusion_matrix(gt_pois, pred_pois)
    # Set values greater than 1 to 1 (for axis 0)
    # confusion_mat[confusion_mat > 0] = 1
    # Set values greater than 1 to 1 (for axis 1)
    # confusion_mat[:, confusion_mat.sum(axis=0) > 1] = 0

    plt.imshow(confusion_mat, cmap=plt.cm.Greens)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xlabel('Predicted (poisoned: 1, clean: 0))')
    plt.ylabel('True (poisoned:1, clean: 0)')

    thresh = confusion_mat.max() / 2.
    cm_color = lambda x: "white" if x > thresh else "black"
    row_sums = np.sum(confusion_mat, axis=1)
    for i in range(len(confusion_mat)):    
        for j in range(len(confusion_mat[i])):
            percentage = round(confusion_mat[i, j]/row_sums[i] * 100, 2)
            text = f"{confusion_mat[i, j]}\n{percentage}%"
            plt.text(j, i, text, horizontalalignment="center", verticalalignment="center", color=cm_color(confusion_mat[i, j]))
            
    if save_path:
        plt.savefig(save_path)

    return confusion_mat