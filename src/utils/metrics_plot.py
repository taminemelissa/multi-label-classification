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
    acc=round(accuracy_score(y_true,y_pred),3)
    pre=round(precision_score(y_true,y_pred,average='micro'),3)
    rec=round(recall_score(y_true,y_pred,average='micro'),3)
    f1=round(f1_score(y_true,y_pred,average='micro'),3)
    hl=round(hamming_loss(y_true,y_pred),3)
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
        
        
def plot_roc_curve_sup(y_test,y_hat_U,y_hat_A,y_hat_UA,title,save=False,name_fig=None):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_hat_UA)
    fpr1, tpr1, threshold1 = metrics.roc_curve(y_test, y_hat_A)
    fpr2, tpr2, threshold2 = metrics.roc_curve(y_test, y_hat_U)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    plt.title(title)
    plt.plot(fpr1, tpr1, 'b', label = 'AUC audio = %0.2f' % roc_auc1)
    plt.plot(fpr2, tpr2, 'r', label = 'AUC usage = %0.2f' % roc_auc2)
    plt.plot(fpr, tpr, 'g', label = 'AUC audio and usage= %0.2f' % roc_auc)
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
    return([roc_auc,roc_auc1,roc_auc2])
    
        
def plot_roc_curve_all(y_test,y_hat_U,y_hat_A,y_hat_UA, algo, save, name_fig1,name_fig2):
    auc_labels_U=[]
    auc_labels_A=[]
    auc_labels_UA=[]
    #conf=multilabel_confusion_matrix(y_test,y_hat_UA)
    l = l=['asian','rnb','reggae','blues', 'pop','dance','folk','arabic-music', 'indie', 'rock', 'soulfunk', 'latin', 'classical', 'k-pop','brazilian', 'metal','rap', 'jazz','electronic','african','country']
    for i in range(len(l)): #pour chaque label, ROC curve
        AUC= plot_roc_curve_sup(y_test[:,i], y_hat_U[:,i],y_hat_A[:,i],y_hat_UA[:,i],"ROC curve "+ l[i],save=save,name_fig=name_fig1 +" "+ l[i])
        auc_labels_U.append(AUC[2])
        auc_labels_A.append(AUC[1])
        auc_labels_UA.append(AUC[0])
        #plot_confusion_matrix(conf[i],title=l[i],classes=[l[i],'Others'],normalize=True,save=save,name_fig=name_fig2) #Confusion matrix juste pour usage et fa
    graph_auc(auc_labels_U,algo+" "+"Usage",save=save,name_fig=name_fig2+" "+"Usage")
    plt.figure()
    graph_auc(auc_labels_A,algo+" "+"Audio",save=save,name_fig=name_fig2+" "+"Audio")
    plt.figure()
    graph_auc(auc_labels_UA,algo+" "+"Usage et audio",save=save,name_fig=name_fig2+" "+"Usage et audio")