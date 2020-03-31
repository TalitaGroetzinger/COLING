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
    bar = Bar()
    for wikihow_instance in list_of_wikihow_instances:
        bar.next()
        wikihow_instance['Source_Line'] = ' '.join(
            wikihow_instance['Source_Tagged'])
        wikihow_instance['Target_Line'] = ' '.join(
            wikihow_instance['Target_Tagged'])
        del wikihow_instance['Source_Tagged']
        del wikihow_instance['Target_Tagged']
    bar.finish()
    return new


def main():
    path_to_file = '../wikihowtools/data/Wikihow_tokenized_v5_cleaned.json'
    print("load data .... ")
    with open(path_to_file, 'r') as json_in:
        content = json.load(json_in)

    # filter the data
    print("clean data")
    #filtered_data = clean_dict(content)
    filtered_data = join_tokens(content)

    # write to new file
    with open("../wikihowtools/data/Wikihow_tokenized_v5_cleaned_tokens_only.json", 'w') as json_out:
        json.dump(filtered_data, json_out)


main()
