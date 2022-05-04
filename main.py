# This is a sample Python script.
import os

import pandas as pd
import click
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    return features_train, features_val, target_train, target_val


if __name__ == '__main__':
    set_the_path = click.prompt('train in current path + data? Y or N', str)
    if set_the_path == 'Y':
        path = os.getcwd() + '\\data\\train.csv'
    elif set_the_path == "N":
        path = click.prompt('please set the path', str)
    else:
        print('you entered the wrong answer')
        exit()
    #path = click.prompt('pliz int the path', str)
    #path = os.getcwd() + '\\data\\train.csv'
    #df = pd.read_csv('train.csv')
    get_dataset(path,42,0.3)

