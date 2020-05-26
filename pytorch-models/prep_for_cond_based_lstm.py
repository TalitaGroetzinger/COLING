import json
from progress.bar import Bar
import pandas as pd
from features_for_pytorch import type_token_ratio


def process_context(context, current_line):
    left_context = context['left']
    right_context = context['right']

    if type(context['left']) == str:
        left_context_plus_current = context['left'] + ' ' + current_line
    else:
        left_context_plus_current = ' '.join(
            context['left']) + ' ' + current_line

    if type(context['right']) == str:
        context = left_context_plus_current + ' ' + context['right']
    else:
        context = left_context_plus_current + ' ' + ' '.join(context['right'])

    return context


def read_data():
    path_to_dir = '../classification-scripts/noun-modifications'
    path_to_test = '{0}/noun-modifications-test-v2-new.json'.format(
        path_to_dir)
    path_to_train = '{0}/noun-modifications-train-v2-new.json'.format(
        path_to_dir)
    path_to_dev = '{0}/noun-modifications-dev-v2-new.json'.format(path_to_dir)

    with open(path_to_test, 'r') as json_in_test:
        test = json.load(json_in_test)

    with open(path_to_train, 'r') as json_in_train:
        train = json.load(json_in_train)

    with open(path_to_dev, 'r') as json_in_dev:
        dev = json.load(json_in_dev)
    return train, dev, test


def get_matches(matches, source=True):
    """
      Returns a list of words that have changed. 
    """
    if source:
        match = [elem[0][0].lower() for elem in matches]
        return ' '.join(match)
    else:
        match = [elem[1][0].lower() for elem in matches]
        return ' '.join(match)


def process_dict(list_of_wikihow_instances, json_to_write_filename):
    collection = []
    bar = Bar("Processing", max=len(list_of_wikihow_instances))
    index_for_source = 0
    index_for_target = 1
    for c, wikihow_instance in enumerate(list_of_wikihow_instances, 1):
        bar.next()
        source_row = {}
        target_row = {}
        # process everything for source
        source_row["Filename"] = wikihow_instance["Filename"]
        source_row["Line"] = wikihow_instance["Source_Line"]
        source_row["Label"] = "0"
        source_row["Noun_mod"] = get_matches(wikihow_instance["PPDB_Matches"])
        source_row["ID"] = index_for_source
        source_row["Context"] = process_context(
            wikihow_instance["Source_Context_5"], wikihow_instance["Source_Line"])

        # get type token ratio
        source_context = process_context(
            wikihow_instance["Source_Context_5"], wikihow_instance["Source_Line"])
        source_row["TTR"] = type_token_ratio(source_context)
        source_row["TTR_Base"] = type_token_ratio(source_context)

        target_row["Line"] = wikihow_instance["Target_Line"]
        target_row["Label"] = "1"
        target_row["Context"] = process_context(
            wikihow_instance["Target_Context_5"], wikihow_instance["Target_Line"])
        target_row["Noun_mod"] = get_matches(
            wikihow_instance["PPDB_Matches"], source=False)

        target_context = process_context(
            wikihow_instance["Target_Context_5"], wikihow_instance["Target_Line"])

        target_context_base = process_context(
            wikihow_instance["Source_Context_5"], wikihow_instance["Target_Line"])

        # get type token_ratio
        target_row["TTR"] = type_token_ratio(target_context)
        target_row["TTR_Base"] = type_token_ratio(target_context_base)

        target_row["ID"] = index_for_target
        collection.append(source_row)
        collection.append(target_row)
        index_for_target = index_for_target + 2
        index_for_source = index_for_source + 2
    bar.finish()
    # convert to dataframe
    df = pd.DataFrame(collection)

    df.to_json(json_to_write_filename, orient='records', lines=True)


def main():
    train_set, dev_set, test_set = read_data()

    process_dict(dev_set, "dev_set_cond_lstm.json")
    #process_dict(test_set, "test_set_cond_lstm.json")
    process_dict(train_set, "train_set_cond_lstm.json")

    # process test set
    # process_dict(test_set, "test_set_pytorch_discourse_sim.json")


main()
