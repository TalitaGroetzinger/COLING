from progress.bar import Bar
import json
import argparse
from nltk.tokenize import word_tokenize
from pprint import pprint
from features import CoherenceFeatures


def get_processed_context(source_context):
    for sent in source_context:
        if '## Timestamp' in sent:
            source_timestamp_index = source_context.index(sent)
            if source_timestamp_index < 5:
                return source_context[source_timestamp_index+1:]
            else:
                return source_context[:source_timestamp_index]
    return source_context


def remove_timestamps(list_of_wikihow_instances):
    """
        This function will be used to remove timestamps from Source_Context_5 and Target_Context_5.
    """
    filtered = []
    bar = Bar('Processing ...', max=len(list_of_wikihow_instances))
    for wikihow_instance in list_of_wikihow_instances:
        bar.next()
        source_context = wikihow_instance['Source_Context_5']
        wikihow_instance['Source_Context_5_Processed'] = get_processed_context(
            source_context)

        # repeat steps for target
        target_context = wikihow_instance['Target_Context_5']
        wikihow_instance['Target_Context_5_Processed'] = get_processed_context(
            target_context)

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
    new_wikihow_instances = remove_timestamps(wikihow_instances)
    #assert len(new_wikihow_instances) == len(wikihow_instances)

    for elem in new_wikihow_instances:
        pprint(elem['Source_Context'])
        print("PROCESSED")
        print("\n")
        print(elem['Source_Context_5_Processed'])
        break

    with open(filename_to_write, "w") as json_out:
        json.dump(new_wikihow_instances, json_out)
