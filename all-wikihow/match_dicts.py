import json
from progress.bar import Bar


def match_dicts(current_set, all_data):
    final_dataset = []
    bar = Bar('Processing ...', max=len(current_set))
    for wikihow_instance in current_set:
        bar.next()
        key_to_find = wikihow_instance['Key']
        try:
            source_line = all_data[key_to_find]['Source_Line_Nr'][0]
            target_line = all_data[key_to_find]['Target_Line_Nr'][-1]
            wikihow_instance['Source_Line_Nr'] = source_line
            wikihow_instance['Target_Line_Nr'] = target_line
            final_dataset.append(wikihow_instance)
        except:
            print("key not found for {0}".format(key_to_find))
    bar.finish()
    return final_dataset


def main():
    with open('splits/wikihow-train.json', 'r') as json_in_subset:
        data = json.load(json_in_subset)

    with open('wikihow_v6_with_numbers_lines.json', 'r') as json_in_all:
        all_data = json.load(json_in_all)

    result = match_dicts(data, all_data)

    with open('splits/wikihow-train-v2.json', 'w') as json_out:
        json.dump(result, json_out)


main()
