import json


def filter_insertions(wikihow_instance_collection):
    collection = []
    for wikihow_instance in wikihow_instance_collection:
        filtered_differences = []
        for elem in wikihow_instance['differences']:
            source = elem[0]
            tag_source = source[1]
            if 'NN' == tag_source or 'NNS' == tag_source or 'NNP' == tag_source:
                filtered_differences.append(elem)
        wikihow_instance['Source_Line_Tagged'] = wikihow_instance['Source_tagged']
        wikihow_instance['Target_Line_Tagged'] = wikihow_instance['Target_Tagged']
        wikihow_instance['Differences'] = filtered_differences
        del wikihow_instance['Source_tagged']
        del wikihow_instance['Target_Tagged']
        del wikihow_instance['differences']
        if wikihow_instance['Differences'] != []:
            collection.append(wikihow_instance)
    return collection


def add_differences(list_with_wikihow_instances):
    data = []
    for wikihow_instance in list_with_wikihow_instances:
        source_tagged = wikihow_instance['Source_tagged']
        target_tagged = wikihow_instance['Target_Tagged']
        differences = []
        for source, target in zip(source_tagged, target_tagged):
            if source[0] != target[0]:
                differences.append([source, target])
        if differences != []:
            wikihow_instance['differences'] = differences
            data.append(wikihow_instance)
    return data


def main():
    with open('../data/wikihow_tokenized_tagged_possible_corrections.json', 'r') as json_in:
        all_corrections = json.load(json_in)
    print("do step 1 ")
    all_corrections = add_differences(all_corrections)
    print("do final step")
    new = filter_insertions(all_corrections)
    print(new[0].keys())

    with open('first-step-same-nouns.json', 'w') as json_in:
        json.dump(new, json_in)


main()
