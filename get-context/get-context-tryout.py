import bz2
import os
import pickle


collection = {}
for filename in os.listdir('dev-files-example'):
    path = "dev-files-example/{0}".format(filename)
    with bz2.open(path, "rt") as bz_file:
        file_in_dict_format = {}
        for counter, line in enumerate(bz_file, 1):
            if line != '\n':
                file_in_dict_format[counter] = line.strip('\n').strip()
    collection[filename] = file_in_dict_format

for key, _ in collection.items():
    print(collection[key])

with open('dev-files-in-dict-format.pickle', 'wb') as pickle_out:
    pickle.dump(collection, pickle_out)
