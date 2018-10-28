import numpy as np
import copy
from sklearn.metrics import fbeta_score


def get_thresholded_predictions(y_oh, threshold=0.5):
    y = copy.copy(y_oh)
    y[y >= threshold] = 1
    y[y != 1] = 0
    return y


def find_naive_threshold_on_fbeta_score(y_true_oh, y_pred_oh, beta=1):
    fb_list = []
    threshold_range = np.linspace(y_pred_oh.min(), y_pred_oh.max(), 100).tolist()
    for thres in threshold_range:
        y_pred = get_thresholded_predictions(y_pred_oh, threshold=thres)
        fb_list.append(fbeta_score(y_true_oh, y_pred, beta=beta, average='samples'))

    thres = threshold_range[int(np.argmax(fb_list))]
    return thres
