import json


def get_xy(wikihow_instances):
    X = []
    Y = []
    for wikihow_instance in wikihow_instances:
        # append source
        source_label = '__label__{0}'.format(str(0))
        Y.append(source_label)
        X.append(wikihow_instance['Source_Context_5_Processed'])
        # append target
        target_label = '__label__{0}'.format(str(1))
        Y.append(target_label)
        X.append(wikihow_instance['Target_Context_5_Processed'])
    assert len(X) == len(Y)
    return X, Y


def main():
    path_to_train = './same-noun-modifications/SAME-NOUN-MODIFICATIONS-TRAIN-5-new.JSON'
    path_to_dev = './same-noun-modifications/SAME-NOUN-MODIFICATIONS-DEV-5-new.JSON'
    path_to_test = './same-noun-modifications/SAME-NOUN-MODIFICATIONS-TEST-5-new.JSON'

    with open(path_to_train, 'r') as json_in:
        wikihow_instances = json.load(json_in)

    X, Y = get_xy(wikihow_instances)
    for elem in X:
        print(len(elem))


main()
