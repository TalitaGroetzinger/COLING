# this script will be used to add the base context to the target line.
# after running this script, it will be necessary to clean the target context by
# removing timestamps

import json
from progress.bar import Bar
import argparse


def add_base_context(list_of_wikihow_instances):
    bar = Bar("Processing .. ", max=len(list_of_wikihow_instances))
    new_wikihow_instances = []
    for wikihow_instance in list_of_wikihow_instances:
        bar.next()
        target_line_in_base_context = []
        # get components from source_context
        source_context = wikihow_instance['Source_Context']
        sentences_before = source_context[0:5]
        sentences_after = source_context[6:]

        # get the target_line
        target_line = wikihow_instance['Target_Line']

        target_line_in_base_context = target_line_in_base_context + sentences_before
        target_line_in_base_context = target_line_in_base_context + \
            [target_line]
        target_line_in_base_context = target_line_in_base_context + sentences_after

        # add to the dictionary
        wikihow_instance['Source_Target_base'] = target_line_in_base_context

        new_wikihow_instances.append(wikihow_instance)
    bar.finish()
    assert len(new_wikihow_instances) == len(list_of_wikihow_instances)
    return new_wikihow_instances


def main():
    # make argparse
    ap = argparse.ArgumentParser(description="Run add_base_context_to_targte")
    ap.add_argument("--input", required=True, type=str,
                    help="File to read")
    ap.add_argument("--output", required=False, type=str,
                    help="File to write")

    args = vars(ap.parse_args())
    path_to_data = args['input']
    filename_to_write = args['output']

    # make a function
    with open(path_to_data, 'r') as json_in:
        content = json.load(json_in)

    new_data = add_base_context(content)

    with open(filename_to_write, 'w') as json_out:
        json.dump(new_data, json_out)


main()
