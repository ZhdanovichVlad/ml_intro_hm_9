from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
import numpy as np
from tqdm import tqdm


def nested_cv_rf (X_train ,y_train, n_cv_outer=10, n_cv_inner=3, random_state=42):
    cv_outer = KFold(n_splits=n_cv_outer, shuffle=True, random_state=random_state)
    # enumerate splits
    outer_results_acc = list()
    outer_results_f1 = list()
    outer_results_roc_auc = list()
    models = list()
    X = X_train
    y = y_train
    for train_ix, test_ix in tqdm(cv_outer.split(X)):
        # split data
        X_train_2, X_test_2 = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train_2, y_test_2 = y.iloc[train_ix], y.iloc[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=n_cv_inner, shuffle=True, random_state=random_state)
        # define the model
        model = RandomForestClassifier(random_state=random_state)
        # define search space
        space = dict()
        space['n_estimators'] = [5, 50, 200]
        space['max_features'] = [2, 4, 6]
        # define search
        scoring = {"AUC": "roc_auc_ovo", "Accuracy": make_scorer(accuracy_score) ,'f1' :'f1_weighted'}
      
        search = GridSearchCV(model, space, scoring=scoring, cv=cv_inner, refit="AUC")
        # execute search
        result = search.fit(X_train_2, y_train_2)

        best_model = result.best_estimator_
        models.append(best_model)

        yhat = best_model.predict(X_test_2)
        yhat_prob = best_model.predict_proba(X_test_2)
        acc = accuracy_score(y_test_2, yhat)
        f1 = f1_score(y_test_2, yhat, average='weighted')
        ras = roc_auc_score(y_test_2, yhat_prob, multi_class='ovo')

        outer_results_acc.append(acc)
        outer_results_f1.append(f1)
        outer_results_roc_auc.append(ras)
        best_number = np.argmax(outer_results_acc)

    return {'model': models[best_number] ,'Accuracy': outer_results_acc[best_number],
            'f1' :outer_results_f1[best_number], 'roc_auc' : outer_results_roc_auc[best_number]}