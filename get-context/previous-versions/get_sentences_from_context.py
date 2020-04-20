import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from progress.bar import Bar
import re
import argparse
import pdb


class WindowError(Exception):
    pass


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_timestamps(list_with_indexes, source_context):
    if any(index == 5 for index in list_with_indexes):
        print("warning, index = 5")
        first_index = list_with_indexes[0]
        second_index = list_with_indexes[-1]
        return source_context[first_index+1:second_index]

        #raise WindowError("Can't process", source_context, list_with_indexes)

    if len(list_with_indexes) == 1:
        index = list_with_indexes[0]
        if index < 5:
            return source_context[index+1:]
        else:
            return source_context[:index]
    else:
        try:
            first_index = max(
                filter(lambda index: index < 5, list_with_indexes))
        except ValueError:
            first_index = 0

        try:
            second_index = min(
                filter(lambda index: index > 5, list_with_indexes))
        except ValueError:
            second_index = -1

        return source_context[first_index+1:second_index]


def no_more_timestamps(context):
    return not any('## Timestamp' in sent for sent in context)


def get_processed_context(source_context):

    timestamp_indexes = []
    for index, sent in enumerate(source_context):
        if '## Timestamp' in sent:
            timestamp_indexes.append(index)
    if timestamp_indexes:
        source_context = remove_timestamps(
            timestamp_indexes, source_context)

        # try:
        #    assert no_more_timestamps(source_context)
        # except AssertionError:
        #    return
    return source_context


def sentence_splitter(document):
    """
      Sentence splitter to deal with bullet items in texts.
    """
    # Tokenize per 'sub sentence list' instead of joining (to keep markdown headers separated)
    unflattened_sentences = (sent_tokenize(sent_item)
                             for sent_item in document)
    # Flatten sentences: (https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists)
    sentences = (
        sentence for sub_sentences in unflattened_sentences for sentence in sub_sentences)
    pattern = re.compile(r"^[0-9]+\.$")
    merged_item_sents = []

    sentence = next(sentences)
    while sentence:
        if re.match(pattern, sentence):
            try:
                next_sentence = next(sentences)
            except StopIteration:
                break
            merged = f"{sentence} {next_sentence}"
            merged_item_sents.append(merged)
        else:
            merged_item_sents.append(sentence)

        try:
            sentence = next(sentences)
        except StopIteration:
            sentence = False
            break
    return merged_item_sents


def get_matching_sent_context(context, sent, windows=[1, 2, 3, 4, 5], use_sent_from_context=False, tokenized=True):
    """
        Use this function to get closest match to a source_line or target_line in a paragraph.
        Tokenized: whether the input sent should be tokenized or not (nesecarry when the sent is a string.)
        use_sent_from_context: if true, then the matched sent will be taken in the final representation.

    """

    sentence_tokenized_document = get_processed_context(context)

    sentence_tokenized_document = sentence_splitter(
        sentence_tokenized_document)

    bleu_scores = []
    sents = []
    if tokenized:
        tokenized_sent = word_tokenize(sent)

    else:
        tokenized_sent = sent

    for elem in sentence_tokenized_document:
        elem = remove_html_tags(elem)
        reference = [word_tokenize(elem)]
        score = sentence_bleu(reference, tokenized_sent)
        bleu_scores.append(score)
        sents.append(elem)
    index_of_max_bleu = bleu_scores.index(max(bleu_scores))
    matched_sent = sents[index_of_max_bleu]

    length_of_sents = len(sents)
    begin_index = max(index_of_max_bleu - 5, 0)
    end_index = (index_of_max_bleu + 5) + 1 \
        if (index_of_max_bleu + 5) < length_of_sents else None

    context = sents[begin_index:end_index]
    return context


def add_filtered_context(list_of_wikihow_instances):
    print(len(list_of_wikihow_instances))
    bar = Bar('Processing ... ', max=len(list_of_wikihow_instances))
    new_wikihow_instances = []
    for index, wikihow_instance in enumerate(list_of_wikihow_instances):
        bar.next()

        try:
            source_context_filtered = get_matching_sent_context(
                wikihow_instance['Source_Context'], wikihow_instance['Source_Line'])

            wikihow_instance['Source_Context_5'] = source_context_filtered
            #  get new context for target"""
            target_context_filtered = get_matching_sent_context(
                wikihow_instance['Target_Context'], wikihow_instance['Target_Line'])
            wikihow_instance['Target_Context_5'] = target_context_filtered

            new_wikihow_instances.append(wikihow_instance)
        except WindowError as error:
            print("===================")
            print("Processing error")
            print("Error: ", error)
            print("Index: ", index)
            print("Instance context: ", wikihow_instance['Source_Context'])
            print("Line : ", wikihow_instance['Source_Line'])
            print("===================")

    bar.finish()
    return new_wikihow_instances


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Train and test classifier")
    ap.add_argument("--input", required=True, type=str,
                    help="File to read")
    ap.add_argument("--output", required=False, type=str,
                    help="File to write")

    args = vars(ap.parse_args())
    filename_to_open = args['input']
    filename_to_write = args['output']
    print("filename: ", filename_to_open)
    print("filename to write: ", filename_to_write)
    with open(filename_to_open, "r") as json_in:
        wikihow_instances = json.load(json_in)
    new_wikihow_instances = add_filtered_context(wikihow_instances)

    try:
        assert len(wikihow_instances) == len(new_wikihow_instances)
    except AssertionError:
        print("Length is not the same: ")
        print("Length file-in: ", len(wikihow_instances))
        print("Length file-out: ", len(new_wikihow_instances))

    with open(filename_to_write, 'w') as json_out:
        json.dump(new_wikihow_instances, json_out)
