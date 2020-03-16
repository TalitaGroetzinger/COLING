import json
import nltk
from nltk import word_tokenize
from progress.bar import Bar

with open('../data/potential-diff-noun-cases.json', 'r') as json_file:
    potential_noun_cases = json.load(json_file)

filtered = []
bar = Bar('Processing', max=len(potential_noun_cases))
for wikihow_instance in potential_noun_cases:
    source = wikihow_instance['All_Versions'][0]
    target = wikihow_instance['All_Versions'][-1]
    # tag alles
    source_tagged = nltk.pos_tag(word_tokenize(source))
    target_tagged = nltk.pos_tag(word_tokenize(target))
    bar.next()
    if len(source_tagged) == len(target_tagged):
        wikihow_instance['Source_Tagged'] = source_tagged
        wikihow_instance['Target_Tagged'] = target_tagged
        del wikihow_instance['Revisions']
        filtered.append(wikihow_instance)
bar.finish()

with open('../data/potential-diff-noun-cases_v2.json', 'w') as json_file:
    json.dump(filtered, json_file)
