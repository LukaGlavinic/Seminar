# import os
# from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# SAVE_DIR_CONF = Path(__file__).parent / 'conf_matrices'

def show_confusion_matrix(extracted_features_Y, predictions, indices_of_poisoned, save_path=None):

    gt_pois = [0] * len(predictions)
    for ind in indices_of_poisoned:
        gt_pois[ind] = 1

    pred_pois = [0] * len(predictions)
    for i, pred in enumerate(predictions):
        if pred != extracted_features_Y[i]:
            pred_pois[i] = 1
    
    confusion_mat = confusion_matrix(gt_pois, pred_pois)

    plt.imshow(confusion_mat, cmap=plt.cm.Greens)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks([0, 1], ['Predicted clean 0', 'Predicted poisoned 1'])
    plt.yticks([0, 1], ['Clean 0', 'Poisoned 1'])

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

# predictions = [1, 2, 3, 4, 5, 6, 4, 8, 9, 0]
# indices_of_poisoned = [3, 7, 4]
# ext_f_y = [1, 3, 5, 6, 2, 0, 9, 8, 7, 5]
# save_path = os.path.join(SAVE_DIR_CONF, 'test_conf_mat')
# show_confusion_matrix(ext_f_y, predictions, indices_of_poisoned, save_path)