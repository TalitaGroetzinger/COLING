import json
from progress.bar import Bar

path = "Wikihow_tokenized_v5.json"


def clean_data(wikihow_collection):
    bar = Bar("Processing ... ", max=len(wikihow_collection))
    cleaned_data = []
    for wikihow_instance in wikihow_collection:
        bar.next()
        # remove unnecessary keys
        del wikihow_instance['Revision_Length']
        del wikihow_instance['Base_Sentence']
        del wikihow_instance['Revisions']
        del wikihow_instance['Source_Tokenized']
        del wikihow_instance['Target_Tokenized']
        del wikihow_instance['Target_Tagged']
        del wikihow_instance['Source_tagged']
        # add two keys that we need: Source Line and Target Line
        # take the first and the last element of "All_Versions"
        wikihow_instance['Source_Line'] = wikihow_instance['All_Versions'][0]
        wikihow_instance['Target_Line'] = wikihow_instance['All_Versions'][-1]
        cleaned_data.append(wikihow_instance)
    bar.finish()
    return cleaned_data


def main():
    with open(path, 'r') as json_in:
        wikihow_collection = json.load(json_in)

    cleaned_data = clean_data(wikihow_collection)

    # write to json
    with open('wikihow_tokenized_v5_lines.json', 'w') as json_out:
        json.dump(cleaned_data, json_out)


main()
