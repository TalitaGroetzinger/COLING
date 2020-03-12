import os


def read_ppdb(path_to_db):
    ppdb_collection = {}
    with open(path_to_db, 'r') as read_file:
        db = read_file.readlines()
    for entry in db:
        entry = entry.strip('\n')
        lhs, phrase, paraphrase, features, alignment, entailment = entry.split(
            "|||")

        lookup_key = "{0}#{1}".format(phrase.strip(), paraphrase.strip())

        ppdb_collection[lookup_key] = {"LHS": lhs.rstrip().lstrip(), "PHRASE": phrase.rstrip().lstrip(),
                                       "PARAPHRASE": paraphrase.rstrip().lstrip(), "ENTAILMENT": entailment.rstrip().lstrip()}
    return ppdb_collection


def main():
    path_to_db = './data/ppdb/ppdb-2.0-s-lexical'
    coll = read_ppdb(path_to_db)
    for key, value in coll.items():
        print(coll, coll[key])


if __name__ == '__main__':
    main()
