import json


def get_list_of_filenames(path_to_file):
    with open(path_to_file, 'r') as json_in:
        list_of_wikihow_instances = json.load(json_in)
    filenames = []
    for wikihow_instance in list_of_wikihow_instances:
        filename = wikihow_instance['Filename']
        filenames.append(filename)
    return list(set(filenames))


def write_file(filename, list_of_filenames):
    print("Write file ..")
    with open(filename, 'w') as json_out:
        for filename in list_of_filenames:
            json_out.write("{0}.bz2 \n".format(filename))


def main():
    path_to_dir = '../classification-scripts/noun-modifications/'

    print("Make file for test")
    path_to_test = '{0}noun-modifications-test-5-new-lines.json'.format(
        path_to_dir)
    print("Make file for train")
    path_to_train = '{0}noun-modifications-train-5-new-lines.json'.format(
        path_to_dir)

    # path to dev is goed gegaan
    print("Make file for dev")
    path_to_dev = '{0}noun-modifications-dev-5-new-lines.json'.format(
        path_to_dir)

    test_filenames = get_list_of_filenames(path_to_test)
    train_filenames = get_list_of_filenames(path_to_train)
    dev_filenames = get_list_of_filenames(path_to_dev)

    write_file('test_files.txt', test_filenames)
    write_file('train_files.txt', train_filenames)
    write_file('dev_files.txt', dev_filenames)


main()
