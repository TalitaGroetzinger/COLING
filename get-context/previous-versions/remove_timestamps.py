from progress.bar import Bar
import json
import argparse
from nltk.tokenize import word_tokenize
from pprint import pprint


def remove_timestamps(list_with_indexes, source_context):
    if len(list_with_indexes) == 1:
        index = list_with_indexes[0]
        if index < 5:
            return source_context[index+1:]
        else:
            return source_context[:index]
    elif len(list_with_indexes) == 2:
        first_index = list_with_indexes[0]
        second_index = list_with_indexes[-1]
        return source_context[first_index+1:second_index]
    else:
        first_index = max(filter(lambda index: index < 5, list_with_indexes))
        second_index = min(filter(lambda index: index > 5, list_with_indexes))

        return source_context[first_index+1:second_index]


def get_processed_context(source_context):
    timestamp_indexes = []
    for index, sent in enumerate(source_context):
        if '## Timestamp' in sent:
            timestamp_indexes.append(index)
            source_context = remove_timestamps(
                timestamp_indexes, source_context)
    return source_context


def remove_timestamps_from_collection(list_of_wikihow_instances):
    """
        This function will be used to remove timestamps from Source_Context_5 and Target_Context_5.
    """
    filtered = []
    bar = Bar('Processing ...', max=len(list_of_wikihow_instances))
    for wikihow_instance in list_of_wikihow_instances:
        bar.next()
        source_context = wikihow_instance['Source_Context_5_new']
        wikihow_instance['Source_Context_5_Processed'] = get_processed_context(
            source_context)

        # repeat steps for target
        # i will just replace this line for now
        target_context = wikihow_instance['Target_Context_5_new']
        wikihow_instance['Target_Context_5_Processed'] = get_processed_context(
            target_context)

        # but still add the other one just in case:
        #target_context_original = wikihow_instance['Target_Context']
        # wikihow_instance['Target_Context_5_Processed_org'] = get_processed_context(
        #    target_context_original)

        filtered.append(wikihow_instance)
    bar.finish()
    return filtered


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Train and test classifier")
    ap.add_argument("--input", required=True, type=str,
                    help="File to read")
    ap.add_argument("--output", required=False, type=str,
                    help="File to write")

    args = vars(ap.parse_args())
    filename_to_open = args['input']
    filename_to_write = args['output']
    print("filename: ", filename_to_open)
    print("filename to write: ", filename_to_write)

    with open(filename_to_open, "r") as json_in:
        wikihow_instances = json.load(json_in)
    new_wikihow_instances = remove_timestamps_from_collection(
        wikihow_instances)

    # check if the length is equal
    assert len(new_wikihow_instances) == len(wikihow_instances)

    for elem in new_wikihow_instances:
        pprint(elem['Source_Context'])
        print("PROCESSED")
        print("\n")
        print(elem['Source_Context_5_Processed'])
        break

    with open(filename_to_write, "w") as json_out:
        json.dump(new_wikihow_instances, json_out)
