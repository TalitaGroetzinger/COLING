import pandas as pd
import json
import random
from nltk.tokenize import word_tokenize


def make_bold(x):
    return "<b> {0} </b>".format(x)


def format_title(title):
    title = title.replace("_", " ").strip('.txt')
    return "<i> How to {0} </i>".format(title)


def get_differences(matches, source_first=True):
    source_matches = [elem[0][0].lower() for elem in matches]
    target_matches = [elem[1][0].lower() for elem in matches]

    differences = []
    if source_first == True:
        # show in source-revise sequence
        for c in range(len(source_matches)):
            output = "{0} -> {1}".format(source_matches[c], target_matches[c])
            if differences != []:
                differences.append(", {0}".format(output))
            else:
                differences.append(output)
    else:
        # show in source-revise sequence
        for c in range(len(source_matches)):
            output = "{1} -> {0}".format(source_matches[c], target_matches[c])
            if differences != []:
                differences.append(", {0}".format(output))
            else:
                differences.append(output)
    return ' '.join(differences)


def highlight_differences(line, matches, index, source=True):
    source_matches = [elem[0][0].lower() for elem in matches]
    target_matches = [elem[1][0].lower() for elem in matches]

    line_tokenized = word_tokenize(line)

    if source:
        matches = source_matches
    else:
        matches = target_matches
    for word in matches:
        try:
            index_of_word = line_tokenized.index(word)
            line_tokenized[index_of_word] = "<b> {0} </b>".format(word)
        except ValueError:
            try:
                index_of_word = line_tokenized.index(word.capitalize())
                line_tokenized[index_of_word] = "<b> {0} </b>".format(
                    word.capitalize())
            except ValueError:
                try:
                    index_of_word = line_tokenized.index(word.upper())
                    line_tokenized[index_of_word] = "<b> {0} </b>".format(
                        word.capitalize())
                except ValueError:
                    print(word, '\t', line, '\t', index)

    return ' '.join(line_tokenized)


def process_context(context, current_line):
    left_context = context['left']
    right_context = context['right']

    if type(context['left']) == str:
        left_context_plus_current = context['left'] + \
            ' ' + make_bold(current_line)
    else:
        left_context_plus_current = ' '.join(
            context['left']) + ' ' + make_bold(current_line)

    if type(context['right']) == str:
        context = left_context_plus_current + ' ' + context['right']
    else:
        context = left_context_plus_current + ' ' + ' '.join(context['right'])

    return context


def read_data(return_dev=True):
    """
      Function to read the dataset
    """
    path_to_dir = '../classification-scripts/noun-modifications'
    if return_dev:
        print("read development set only")
        path_to_dev = '{0}/noun-modifications-dev-v2-new.json'.format(
            path_to_dir)
        with open(path_to_dev, 'r') as json_in_dev:
            dev = json.load(json_in_dev)
        return dev
    else:
        print("read train, dev and test")
        path_to_test = '{0}/noun-modifications-test-v2-new.json'.format(
            path_to_dir)
        path_to_train = '{0}/noun-modifications-train-v2-new.json'.format(
            path_to_dir)

    with open(path_to_test, 'r') as json_in_test:
        test = json.load(json_in_test)

    with open(path_to_train, 'r') as json_in_train:
        train = json.load(json_in_train)

    with open(path_to_dev, 'r') as json_in_dev:
        dev = json.load(json_in_dev)
    return train, dev, test


def randomize_source_base(base_line, revised_line, base_line_in_base_context, revised_line_in_base_context, differences, index):
    # pick a number to decide which components will be mentioned first: 0 (base) or 1 (revised)
    element_that_will_be_presented_first = random.choice([0, 1])

    # means that the source is the first one:
    base_line = highlight_differences(base_line, differences, index)
    revised_line = highlight_differences(
        revised_line, differences, index, source=False)
    if element_that_will_be_presented_first == 0:
        elements = {"Line1": base_line, "Context1": base_line_in_base_context,
                    "Line2": revised_line, "Context2": revised_line_in_base_context, "Line1BaseOrSource": "base",
                    "Differences": get_differences(differences)}
    else:

        elements = {"Line2": base_line, "Context2": base_line_in_base_context,
                    "Line1": revised_line, "Context1": revised_line_in_base_context, "Line1BaseOrSource": "revised",
                    "Differences": get_differences(differences, source_first=False)}

    return elements


def process_dict(list_of_wikihow_instances, sample_size=500):
    collection = []
    index_for_source = 0
    index_for_target = 1
    row_index = 0
    for c, wikihow_instance in enumerate(list_of_wikihow_instances, 1):
        row = {}
        # process everything for source
        row["Title"] = format_title(wikihow_instance["Filename"])
        row["Batch_ID"] = "Source: {0} Target: {1}".format(
            index_for_source, index_for_target)

        # get the elements necessary for the CSV
        base_line = wikihow_instance["Source_Line"]
        base_line_in_base_context = revised_line_in_base_context = process_context(
            wikihow_instance["Source_Context_5"], wikihow_instance["Source_Line"])

        revised_line = wikihow_instance["Target_Line"]
        revised_line_in_base_context = process_context(
            wikihow_instance["Source_Context_5"], wikihow_instance["Target_Line"])

        elements_for_csv = randomize_source_base(
            base_line, revised_line, base_line_in_base_context, revised_line_in_base_context, wikihow_instance["Differences"], row_index)

        row["Line1"] = elements_for_csv["Line1"]
        row["Line2"] = elements_for_csv["Line2"]
        row["Context1"] = elements_for_csv["Context1"]
        row["Context2"] = elements_for_csv["Context2"]
        row["Info"] = elements_for_csv["Line1BaseOrSource"]
        row["Differences"] = elements_for_csv["Differences"]

        collection.append(row)
        index_for_target = index_for_target + 2
        index_for_source = index_for_source + 2
        row_index = row_index + 1

    df = pd.DataFrame(collection)
    subset = df.sample(n=len(df), random_state=1).head(500)
    print(subset)
    subset.to_csv('annotation-subset.csv', index=False)


def main():
    dev = read_data()
    process_dict(dev)


main()
