from src.modelisation.utils import *
import numpy as np
from sklearn.dummy import DummyClassifier

def random_classifier(X_train, y_train):
    
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(X_train, y_train)
    return dummy_clf