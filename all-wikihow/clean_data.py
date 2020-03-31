import json
from progress.bar import Bar


def clean_dict(list_of_wikihow_instances):
    filtered_data = []
    bar = Bar("Processing ...", max=len(list_of_wikihow_instances))
    for elem in list_of_wikihow_instances:
        bar.next()
        # delete elements which are unnecessary
        del elem['Base_Sentence']
        del elem['Revisions']
        del elem['Source_Tokenized']
        del elem['Target_Tokenized']
        del elem['All_Versions']

        # rename source_tagged
        elem['Source_Tagged'] = elem['Source_tagged']
        del elem['Source_tagged']
        filtered_data.append(elem)
    bar.finish()
    try:
        assert len(filtered_data) == len(list_of_wikihow_instances)
    except AssertionError:
        print("length is unequal: filtered: {0} \t unfiltered: {1} ".format(
            len(filtered_data), len(list_of_wikihow_instances)))
    return filtered_data


def join_tokens(list_of_wikihow_instances):
    new = []
    bar = Bar('Processing ...', max=len(list_of_wikihow_instances))
    for wikihow_instance in list_of_wikihow_instances:
        bar.next()

        source_tokens = [pair[0] for pair in wikihow_instance['Source_Tagged']]
        target_tokens = [pair[0] for pair in wikihow_instance['Target_Tagged']]
        wikihow_instance['Source_Line'] = source_tokens
        wikihow_instance['Target_Line'] = target_tokens

        del wikihow_instance['Source_Tagged']
        del wikihow_instance['Target_Tagged']
        new.append(wikihow_instance)
    bar.finish()
    return new


def make_splits(list_of_wikihow_instances):
    bar = Bar('Processing ', max=len(list_of_wikihow_instances))
    dev = []
    test = []
    train = []
    for wikihow_instance in list_of_wikihow_instances:
        bar.next()
        if wikihow_instance['Loc_in_splits'] == 'DEV':
            dev.append(wikihow_instance)
        elif wikihow_instance['Loc_in_splits'] == 'TEST':
            test.append(wikihow_instance)
        else:
            train.append(wikihow_instance)
    bar.finish()
    return dev, test, train


def main():
    path_to_file = '../data/Wikihow_tokenized_v5_cleaned_splits.json'
    print("load data .... ")
    with open(path_to_file, 'r') as json_in:
        content = json.load(json_in)

    # filter the data
    print("clean data")
    filtered_data = join_tokens(content)

    # write to new file
    print("write to new file ../wikihowtools/data/Wikihow_tokenized_v5_cleaned_splits_tokens_only.json")
    with open("../wikihowtools/data/Wikihow_tokenized_v5_cleaned_splits_tokens_only.json", 'w') as json_out:
        json.dump(filtered_data, json_out)


main()
