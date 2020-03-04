import os


def read_ppdb(path_to_db):
    ppdb_collection = []
    with open(path_to_db, 'r') as read_file:
        db = read_file.readlines()
    for entry in db:
        entry = entry.strip('\n')
        lhs, phrase, paraphrase, features, alignment, entailment = entry.split(
            "|||")

        entry_in_dict = {"LHS": lhs.rstrip().lstrip(), "PHRASE": phrase.rstrip().lstrip(),
                         "PARAPHRASE": paraphrase.rstrip().lstrip(), "ENTAILMENT": entailment.rstrip().lstrip()}
        ppdb_collection.append(entry_in_dict)
    return ppdb_collection


def main():
    path_to_db = './data/ppdb/ppdb-2.0-s-lexical'
    read_ppdb(path_to_db)


if __name__ == '__main__':
    main()
