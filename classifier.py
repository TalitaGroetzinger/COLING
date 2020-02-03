from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def get_scores(Ytrue, Ypredicted):
    report = classification_report(Ytrue, Ypredicted)
    accuracy = accuracy_score(Ytrue, Ypredicted)
    print("The accuracy is {0}".format(accuracy))
    print(report)


def main():
    X = ""
    Y = ""
    classifier = MultinomialNB()
    model = classifier.fit(X, Y)
    # use predict_proba here
    # -----------------------


if __name__ == '__main__':
    main()
