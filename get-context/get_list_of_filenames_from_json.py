import json


def write_to_file(filename, list_with_filenames):
    with open(filename, 'w') as file_out:
        for elem in list_with_filenames:
            line_to_write = "{0}.bz2{1}".format(elem, '\n')
            file_out.write(line_to_write)
    print("finished")


def main():
    with open('../classification-scripts/same-noun-modifications/same-noun-modifications-line-nr.json', 'r') as json_in:
        same_noun_modifications = json.load(json_in)

    dev_files = []
    test_files = []
    train_files = []
    for elem in same_noun_modifications:
        if elem['Loc_in_splits'] == 'DEV':
            dev_files.append(elem['Filename'])
        elif elem['Loc_in_splits'] == 'TRAIN':
            train_files.append(elem['Filename'])
        else:
            test_files.append(elem['Filename'])

    write_to_file("dev_files_same_noun.txt", list(set(dev_files)))
    write_to_file("test_files_same_noun.txt", list(set(test_files)))
    write_to_file("train_file_same_noun.txt", list(set(train_files)))


main()
