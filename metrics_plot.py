from sklearn import metrics
from math import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
import numpy as np
import seaborn as sns


#return la valeur de l'auc en plus
def plot_roc_curve(y_test,y_hat,title,save=False,name_fig=None):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_hat)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if not save:
        plt.show()
    else:
        plt.savefig(name_fig)
        plt.clf()
    return(roc_auc)

import itertools

""" Fonction qui permet de plot la confusion matrix normalisée ou pas, choisir de save ou pas, title correspond ici au titre de la fig
et pas au nom du fichier, la matrice est plot dans le sens usuel ie [TP,FN]
                                                                    [FP,TN] """
def plot_confusion_matrix(cm, title, classes,cmap=plt.cm.Blues, normalize=False, save=False,name_fig=None):
    cm2=cm

    if normalize:
        cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
        for i in range (2):
            for j in range (2):
                cm2[i][j]=round(cm2[i][j],2)
    thresh = cm2.max() / 2.
    tmp=cm2[0][0]
    cm2[0][0]=cm2[1][1]
    cm2[1][1]=tmp
    tmp=cm2[0][1]
    cm2[0][1]=cm2[1][0]
    cm2[1][0]=tmp
    g=sns.heatmap(cm2,annot=True, fmt='.2%',cmap='Blues')
    if normalize: 
        s='Normalized confusion matrix '+title
        g.set_title(s)
    else:
        s='Not Normalized confusion matrix '+title
        g.set_title(s)
    g.set_xlabel('Predicted Label')
    g.set_ylabel('True Label')
    g.set_xticklabels(classes)
    g.set_yticklabels(classes)
    if save :
        plt.savefig(name_fig)
    else:
         plt.show()
    plt.clf()
    return()

def all_metrics(y_true,y_pred):
    acc=accuracy_score(y_true,y_pred)
    pre=precision_score(y_true,y_pred,average='micro')
    rec=recall_score(y_true,y_pred,average='micro')
    f1=f1_score(y_true,y_pred,average='micro')
    hl=hamming_loss(y_true,y_pred)
    r=[acc,pre,rec,f1,hl]
    return(r)

def graph_auc(auc_values,method,save=False,name_fig=None):
    l=['asian','rnb','reggae','blues', 'pop','dance','folk','arabic-music', 'indie', 'rock', 'soulfunk', 'latin', 'classical', 'k-pop','brazilian', 'metal','rap', 'jazz','electronic','african','country']
    x=list(np.arange(1,22))
    plt.bar(x,auc_values)
    plt.xlabel('Label')
    plt.xticks(x,l,rotation=90)
    plt.ylabel('AUC')
    plt.title('AUC for each label '+method)
    if save:
        plt.savefig(name_fig)
    else:
        plt.show()