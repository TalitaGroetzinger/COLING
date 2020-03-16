import json

with open('../data/potential-diff-noun-cases.json', 'r') as json_file:
    potential_noun_cases = json.load(json_file)

with open('../classification-scripts/classification-data/DIFF-NOUN-MODIFICATIONS.json', 'r') as json_file2:
    diff_noun_files = json.load(json_file2)


for wikihow_instance1, wikihow_instance2 in zip(potential_noun_cases, diff_noun_files):
    all_versions_source = wikihow_instance1['All_Versions'][0]
    source_untokenized = ' '.join(
        [pair[0] for pair in wikihow_instance2['Source_tagged']])
    print(all_versions_source)
    if all_versions_source == source_untokenized:
        print(wikihow_instance1)
        print("----------------------------------------")
