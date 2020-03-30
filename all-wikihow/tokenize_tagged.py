import json
from progress.bar import Bar

path = '../data/Wikihow_tokenized_v5_cleaned_splits.json'
#path = '../data/subset-splits.json'
with open(path, 'r') as json_in:
    content = json.load(json_in)

new = []
bar = Bar('Processing ...', max=len(content))
for wikihow_instance in content:
    bar.next()
    wikihow_instance['Source_Tokens'] = [pair[0]
                                         for pair in wikihow_instance['Source_Tagged']]
    wikihow_instance['Target_Tokens'] = [pair[0]
                                         for pair in wikihow_instance['Target_Tagged']]
    del wikihow_instance['Source_Tagged']
    del wikihow_instance['Target_Tagged']
    new.append(wikihow_instance)
bar.finish()

filename_to_write = "../data/Wikihow_tokenized_v5_cleaned_splits_tokens_only.json"
with open(filename_to_write, 'w') as json_out:
    json.dump(new, json_out)
