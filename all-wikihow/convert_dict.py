import json
from progress.bar import Bar


def convert_dict(data):
    collection = {}
    bar = Bar('Processing ...', max=len(data))
    for wikihow_instance in data:
        bar.next()
        row = {}
        row['Filename'] = wikihow_instance['Filename']
        row['Source_Line_Nr'] = wikihow_instance['Source_Line_Nr']
        row['Target_Line_Nr'] = wikihow_instance['Target_Line_Nr']

        collection[wikihow_instance['key']] = row
    bar.finish()
    return collection


def main():
    with open('../../wiki-how-scripts/tsv-to-json/wikihow_v6_with_numbers.json') as json_in:
        data = json.load(json_in)

    print(len(data))
    new_dict = convert_dict(data)

    with open('wikihow_v6_with_numbers_lines.json', 'w') as json_out:
        json.dump(new_dict, json_out)


main()
