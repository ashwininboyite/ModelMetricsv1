# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:26:59 2021

@author: aboyite
"""

import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

#modelop.init
def begin():
    pass

#modelop.score
def action(datum):
    yield datum

#modelop.score
def compute_metrics(data):
    """
    A function to evaluate a classification model
    
    param: y: true (actual) labels
    param: y_preds: predicted labels (as scored by model)
    
    return: mutiple classification performance metrics
    """
    print(data)
    data = pd.DataFrame(data)
    data.columns = ['prediction','actual']
    
    y = pd.to_numeric(data['prediction'])
    y_preds = pd.to_numeric(data['actual'])
    
    metrics = {  
    "accuracy":accuracy_score(y, y_preds),
    "precision":precision_score(y, y_preds),
    "recall":recall_score(y, y_preds),
    "f1_score":f1_score(y, y_preds),
    "f2_score":fbeta_score(y, y_preds, beta=2)
    }
    print(metrics)
    
    yield metrics 
