import json
from progress.bar import Bar


path_to_train = 'wikihow-test.json'

with open(path_to_train) as json_in:
    content = json.load(json_in)


list_of_files = list(set([wikihow_instance['Filename']
                          for wikihow_instance in content]))
with open('test_files.txt', 'w') as file_out:
    bar = Bar('Processing ...', max=len(list_of_files))
    for filename in list_of_files:
        bar.next()
        line_to_write = '{0}.bz2{1}'.format(filename, '\n')
        file_out.write(line_to_write)
    bar.finish()
