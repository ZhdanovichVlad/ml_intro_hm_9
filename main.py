# This is a sample Python script.
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import click
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import get_dataset
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import nested_cross_validation_with_random_forest
import pipline
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



if __name__ == '__main__':
    set_the_path = click.prompt('train in current path + data? Y or N', str)
    if set_the_path == 'Y':
        path = os.getcwd() + '\\data\\train.csv'
    elif set_the_path == "N":
        path = click.prompt('please set the path', str)
    else:
        print('you entered the wrong answer')
        exit()
    random_state = int(click.prompt('please enter random state (int) ', int))
    test_split_ratio = float(click.prompt('please enter test_split_ratio (float)',float))
    X_train, X_test, y_train, y_test = get_dataset.get_dataset(path,random_state,test_split_ratio)

    n_cv_outer = int(click.prompt('please set the number of cv for outer cross validation', str))
    n_cv_in = int(click.prompt('please set the number of cv for inner cross validation', str))
    model_nsvwrf = nested_cross_validation_with_random_forest.nested_cv_rf(X_train,y_train,n_cv_outer,n_cv_in,
                                                                           random_state)

    #mlflow.set_tracking_uri('http://localhost:5000')

    with mlflow.start_run():
        model_nsvwrf = model_nsvwrf['model']
        ansv_model_nsvwrf = model_nsvwrf.predict(X_test)
        ansv_proba_model_nsvwrf = model_nsvwrf.predict_proba(X_test)
        acc = accuracy_score(y_test, ansv_model_nsvwrf)
        f1 = f1_score(y_test, ansv_model_nsvwrf, average='weighted')
        ras = roc_auc_score(y_test, ansv_proba_model_nsvwrf, multi_class='ovo')
        mlflow.sklearn.log_model(model_nsvwrf, 'nested_cross_validation_with_random_forest')
        mlflow.log_metric('accuracy',acc)
        mlflow.log_metric('f1', f1)
        mlflow.log_metric('roc_auc_score', ras)








    #print(roc_auc_score(np.array(y_test), np.array(y_test), multi_class='ovo', average='weighted'))
    #model = RandomForestClassifier(random_state=random_state)
    # define search space
    #space = dict()
    #space['n_estimators'] = [10, 100, 500]
    #space['max_features'] = [2, 4, 6]
    # define search
    #search = GridSearchCV(model, space, scoring=('r2','accuracy','balanced_accuracy'), n_jobs=1, cv=5, refit=True)
    # configure the cross-validation procedure
    #cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
    # execute the nested cross-validation
    #scores = cross_val_score(search, X_train, y_train, scoring='accuracy', cv=cv_outer, n_jobs=-1)
    # report performance
    #print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))



    #print(mse)
    #print(result)
