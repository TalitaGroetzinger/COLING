import json

with open('../classification-scripts/classification-data/DIFF-NOUN-MODIFICATIONS.json', 'r') as json_in:
    diff_noun_modifications = json.load(json_in)


with open('diff_noun_modifications.tsv', 'w') as tsv_file:
    header = "{0}\t{1}\t{2}".format(
        'Filename', 'Source_Tagged', 'Target_Tagged')
    tsv_file.write(header)
    for elem in diff_noun_modifications:
        line_to_write = "{0}\t{1}\t{2}\n".format(
            elem['Filename'], elem['Source_tagged'], elem['Target_Tagged'])
        tsv_file.write(line_to_write)
