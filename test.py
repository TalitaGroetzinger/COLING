import json
from progress.bar import Bar


def check_corrections(list_of_wikihow_instances):
    bar = Bar('Processing', max=len(list_of_wikihow_instances))
    for wikihow_instance in list_of_wikihow_instances:
        tags_source = [pair[1] for pair in wikihow_instance['Source_tagged']]
        tags_target = [pair[1] for pair in wikihow_instance['Target_Tagged']]
        assert tags_source != tags_target
        assert len(tags_source) == len(tags_target)
        bar.next()
    bar.finish()


with open('./data/wikihow_tokenized_tagged_possible_corrections_v2.json', 'r') as json_file:
    list_of_wikihow_instances = json.load(json_file)

print(len(list_of_wikihow_instances))
check_corrections(list_of_wikihow_instances)
