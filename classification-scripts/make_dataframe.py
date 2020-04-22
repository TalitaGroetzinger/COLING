# this script can be used to fit different columns to the vectorizer
import json
from collections import Counter
import pandas as pd


def regroup_context(context):
    merged_context = [value if type(value) == str else ' '.join(value)
                      for key, value in context.items()]
    return ' '.join(merged_context)


def get_paths():
    path_to_dir = './noun-modifications'
    path_to_test = '{0}/noun-modifications-test-v2-new.json'.format(
        path_to_dir)
    path_to_train = '{0}/noun-modifications-train-v2-new.json'.format(
        path_to_dir)
    path_to_dev = '{0}/noun-modifications-dev-v2-new.json'.format(path_to_dir)
    return path_to_train, path_to_dev, path_to_test


def make_df(json_file):
    df_dict = {"X_Line": [], "X_Context": [], "Y": []}
    for wikihow_instance in json_file:
        # add the source components
        df_dict["X_Line"].append(wikihow_instance['Source_Line'])
        df_dict["X_Context"].append(regroup_context(
            wikihow_instance['Source_Context_5']))
        df_dict["Y"].append(0)
        # add the target components
        df_dict["X_Line"].append(wikihow_instance['Target_Line'])
        df_dict["X_Context"].append(regroup_context(
            wikihow_instance['Target_Context_5']))
        df_dict["Y"].append(1)

    df = pd.DataFrame.from_dict(df_dict)
    assert len(df) == len(json_file)*2
    return df


def main():
    path_to_train, path_to_dev, path_to_test = get_paths()
    with open(path_to_dev, 'r') as json_in:
        dev = json.load(json_in)
    with open(path_to_train, 'r') as json_in:
        train = json.load(json_in)
    with open(path_to_test, 'r') as json_in:
        test = json.load(json_in)

    dev_df = make_df(dev)
    dev_df.to_pickle("./noun-modifications/dev-df.pickle")

    # do same for train
    train_df = make_df(train)
    train_df.to_pickle("./noun-modifications/train-df.pickle")
    # do same for test
    test_df = make_df(test)
    test_df.to_pickle("./noun-modifications/test-df.pickle")


main()
