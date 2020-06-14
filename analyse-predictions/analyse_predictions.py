import pickle
import numpy as np
import matplotlib.pyplot as plt


def descriptives(list_with_preds):
    print("Mean: {0}".format(np.mean(list_with_preds)))
    print("Max: {0}".format(np.max(list_with_preds)))
    print("Min: {0}".format(np.min(list_with_preds)))
    print("Std: {0}".format(np.std(list_with_preds)))


def make_plot(list_with_preds, list_with_batchids):
    colors = (0, 0, 0)
    area = np.pi*3
    plt.scatter(list_with_preds, list_with_batchids,
                s=area, c=colors, alpha=0.5)
    plt.title(
        'Predictions from the best model on the development set')
    plt.xlabel('Predictions')
    plt.ylabel('Instance ID')
    plt.savefig('scatter_plot.png', dpi=1200, format='png')


def compute_predictions(predictions):
    """
      Use this function in case we want to check the accuracy.
    """
    good = 0
    bad = 0
    for source, target in zip(predictions[::2], predictions[1::2]):
        source_pred = source[0]
        target_pred = target[0]

        if source_pred < target_pred:
            good += 1
        else:
            bad += 1
    print(good/(good+bad))


def compute_positives_negatives(predictions):
    # recognise revised as better
    true_positives = 0
    # recognise base as better
    false_positives = 0
    false_negatives = 0
    for source, target in zip(predictions[::2], predictions[1::2]):
        source_pred = source[0]
        target_pred = target[0]

        if round(source_pred, 0) == round(target_pred, 0):
            false_negatives += 1
        else:
            if source_pred < target_pred:
                true_positives += 1
            else:
                false_positives += 1
    print("TP", true_positives)
    print("FP", false_positives)
    print("FN", false_negatives)


def compute_normal_accuracy(predictions, gold):
    rounded_predictions = [float(round(pred, 0)) for pred in predictions]
    correct = 0
    for ypred, goldpred in zip(rounded_predictions, gold):
        if ypred == goldpred:
            correct += 1
    print(correct/len(predictions))


def get_intervals(list_of_predictions):
    class_1 = []
    class_2 = []
    class_3 = []
    for prediction in list_of_predictions:
        if prediction <= 0.4:
            class_1.append(prediction)
        elif prediction > 0.4 and prediction <= 0.6:
            print(prediction)
            class_2.append(prediction)
        else:
            class_3.append(prediction)
    print("Cases smaller or equal to 0.4: {0} {1}".format(
        len(class_1), len(class_1)/len(list_of_predictions)))
    print("Cases bigger than 0.4 but smaller than 0.6: {0} {1}".format(
        len(class_2), len(class_2)/len(list_of_predictions)))
    print("Cases bigger or equal to 0.6: {0} {1}".format(
        len(class_3), len(class_3)/len(list_of_predictions)))


def check_differences(predictions):
    differences = []
    for source, target in zip(predictions[::2], predictions[1::2]):
        source_pred = source[0]
        target_pred = target[0]
        diff = abs(target_pred-source_pred)
        differences.append(diff)
    return differences


def check_predictive_intervals(differences):
    smaller_than_1 = []
    first_interval = []
    second_interval = []
    third_interval = []
    fourth_interval = []
    fifth_interval = []
    biggest_interval = []

    for difference in differences:
        if difference < 0.1:
            smaller_than_1.append(difference)
        elif difference >= 0.1 and difference < 0.2:
            print(difference)
            first_interval.append(difference)
        elif difference >= 0.2 and difference < 0.3:
            second_interval.append(difference)
        elif difference >= 0.3 and difference < 0.4:
            third_interval.append(difference)
        elif difference >= 0.4 and difference < 0.5:
            fourth_interval.append(difference)
        elif difference >= 0.6 and difference < 0.7:
            fifth_interval.append(difference)
        else:
            biggest_interval.append(difference)

    return first_interval, second_interval, third_interval, fourth_interval, fifth_interval, biggest_interval


def main():
    with open('predictions_of_best_model.pickle', 'rb') as pickle_in:
        predictions = pickle.load(pickle_in)

    predictions_only = []
    batch_ids = []
    gold_pred = []
    for elem in predictions:
        pred, gold, batch_id = elem
        predictions_only.append(pred)
        batch_ids.append(batch_id)
        gold_pred.append(gold)

    # descriptives(predictions_only)
    # get_intervals(predictions_only)
    # print(len(predictions_only))

    # make_plot(predictions_only, batch_ids)
    differences = check_differences(predictions)
    # descriptives(differences)
    """
    first_interval, second_interval, third_interval, fourth_interval, fifth_interval, biggest_interval = check_predictive_intervals(
        differences[0:10])

    print(differences[0:10])

    print(len(first_interval))
    print(len(second_interval))
    print(len(third_interval))
    print(len(fourth_interval))
    print(len(fifth_interval))
    print(len(biggest_interval))
    """
    #compute_normal_accuracy(predictions_only, gold_pred)

    print(predictions[0:10])
    compute_positives_negatives(predictions)


main()
