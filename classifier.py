from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from wikihowtools.add_linguistic_info import read_json, get_source_revision_pair


def get_data(collection):
    # make x and y for first version
    X_first_version = [elem['first_version'] for elem in collection]
    Y_first_version = [0 for i in range(len(X_first_version))]
    test_length_documents_labels(X_first_version, Y_first_version)

    # make x and y for revised version
    X_final_version = [elem['final_version'] for elem in collection]
    Y_final_version = [1 for i in range(len(X_final_version))]
    test_length_documents_labels(X_final_version, Y_final_version)

    # merge the lists
    X = X_first_version + X_final_version
    Y = X_final_version + Y_final_version
    test_length_documents_labels(X, Y)
    return X, Y


def get_scores(Ytrue, Ypredicted):
    report = classification_report(Ytrue, Ypredicted)
    accuracy = accuracy_score(Ytrue, Ypredicted)
    print("The accuracy is {0}".format(accuracy))
    print(report)


def test_length_documents_labels(labels, docs):
    try:
        assert len(labels) == len(docs)
    except AssertionError:
        print("Length is unequal:")
        print("Number of Documents:", len(docs))
        print("Number of labels:", len(labels))


def main():
    # load the json file
    path = '../wiki-how-scripts/tsv-to-json/json-files/wikihow_v6_v3.json'
    wikihow_json = read_json(path)
    # return collection of dictionaries containing the first and final version of each article.
    collection = get_source_revision_pair(wikihow_json)
    X, Y = get_data(collection[0:20])
    print(X[0])
    print(Y[0])

    #X = ""
    #Y = ""
    #classifier = MultinomialNB()
    #model = classifier.fit(X, Y)
    # use predict_proba here
    # -----------------------


if __name__ == '__main__':
    main()
