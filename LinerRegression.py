import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import pandas as pd
import click
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import get_dataset
import pipline
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error

def Liner_Model(X_train,y_train,random_state=42, n_cv = 5):
    model = pipline.create_pipeline(True,100,1.0,random_state)
    result = cross_validate(model, X_train, y_train, cv=n_cv, return_train_score=True,
                            scoring=('r2', 'neg_mean_squared_error', 'accuracy'), return_estimator=True)
    number_of_model = np.argmax(result['test_accuracy'])
    return result['estimator'][number_of_model]
