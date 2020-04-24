import json

with open('train-dict-format.json', 'r') as json_in:
    wikihow_articles = json.load(json_in)


def get_line_and_file(filename, line_nr, collection):
    # get current_line
    try:
        current_line = collection[filename][line_nr]
    except KeyError:
        print("can not find file {0}".format(filename))
    # make a list of all line numbers
    sentence_nrs = [key for key, _ in collection[filename].items()]
    list_positions = [i for i in range(len(sentence_nrs))]
    # get the index of the line before and after
    window_range = [1, 2, 3, 4, 5]
    sents_before_current = []
    sents_after_current = []

    for window in window_range:
        previous_line_index = sentence_nrs.index(line_nr)-window
        next_line_index = sentence_nrs.index(line_nr)+window
        if previous_line_index in list_positions:
            previous_line_pos = sentence_nrs[previous_line_index]
            previous_line = collection[filename][previous_line_pos]
            sents_before_current.append(previous_line)

        if next_line_index in list_positions:
            next_line_pos = sentence_nrs[next_line_index]
            next_line = collection[filename][next_line_pos]
            sents_after_current.append(next_line)

    sents_before_current.reverse()

    return {
        "left": ' '.join(sents_before_current),
        "current": current_line,
        "right": ' '.join(sents_after_current),
    }


def main():
    #path_to_data = '../classification-scripts/noun-modifications/noun-modifications-test-5-new-lines.json'
    path_to_data = 'noun-modifications-train-5-new-lines.json'
    with open(path_to_data, 'r') as json_in:
        list_of_wikihow_instances = json.load(json_in)

    collection = wikihow_articles

    new_instances = []
    for wikihow_instance in list_of_wikihow_instances:
        source_line_nr = wikihow_instance['Source_Line_Nr'][0]
        target_line_nr = wikihow_instance['Target_Line_Nr'][-1]

        filename = wikihow_instance['Filename']
        filename = wikihow_instance['Filename']  # + bz2

        source_line_nr_content = get_line_and_file(
            filename, str(source_line_nr), collection)
        print(source_line_nr_content)
        wikihow_instance['Source_Context_New'] = source_line_nr_content

        target_line_nr_content = get_line_and_file(
            filename, str(target_line_nr), collection)
        wikihow_instance['Target_Context_New'] = target_line_nr_content
        new_instances.append(wikihow_instance)

    assert len(new_instances) == len(list_of_wikihow_instances)

    with open('noun-modifications-train-v1-length.json', 'w') as json_out:
        json.dump(new_instances, json_out)


main()
