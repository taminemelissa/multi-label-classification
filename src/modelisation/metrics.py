import numpy as np
from sklearn import metrics


def ROC(y, y_predicted, model_name):
    fpr, tpr, thresholds = metrics.roc_curve(y, y_predicted, pos_label=2)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
    display.plot()
    plt.show()  
