import os
import click
import LinerRegression
import get_dataset
import mlflow
import nested_cross_validation_with_random_forest
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



if __name__ == '__main__':
    set_the_path = click.prompt('Train in current path + data? Y or N', str)
    if set_the_path == 'Y':
        path = os.getcwd() + '\\data\\train.csv'
    elif set_the_path == "N":
        path = click.prompt('Please set the path', str)
    else:
        print('You entered the wrong answer')
        exit()

    random_state = int(click.prompt('Please enter random state (int) ', int))
    test_split_ratio = float(click.prompt('Please enter test_split_ratio (float)',float))
    X_train, X_test, y_train, y_test = get_dataset.get_dataset(path,random_state,test_split_ratio)

    print('Please select the model.')
    print('Enter 1 if you want to select nested_cross_validation_with_random_forest')
    print('Enter 2 if you want to select liner regression')
    n_model = int(click.prompt('Enter the number', int))
    if n_model == 1:
        n_cv_outer = int(click.prompt('Please set the number of cv for outer cross validation', int))
        n_cv_in = int(click.prompt('Please set the number of cv for inner cross validation', int))
        model_nsvwrf = nested_cross_validation_with_random_forest.nested_cv_rf(X_train,y_train,n_cv_outer,n_cv_in,
                                                                           random_state)
        model = model_nsvwrf['model']
        model_name = 'nested_cross_validation_with_random_forest'

    if n_model == 2:
        n_cv_in = int(click.prompt('Please set the number of cv for cross validation', int))
        model = LinerRegression.Liner_Model(X_train,y_train,random_state,n_cv_in)
        model_name = 'Liner Regression with cross validation'




    with mlflow.start_run():
        ansv_model_nsvwrf = model.predict(X_test)
        ansv_proba_model_nsvwrf = model.predict_proba(X_test)
        acc = accuracy_score(y_test, ansv_model_nsvwrf)
        f1 = f1_score(y_test, ansv_model_nsvwrf, average='weighted')
        ras = roc_auc_score(y_test, ansv_proba_model_nsvwrf, multi_class='ovo')
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_metric('accuracy',acc)
        mlflow.log_metric('f1', f1)
        mlflow.log_metric('roc_auc_score', ras)
        click.echo(f"Model is saved.")
