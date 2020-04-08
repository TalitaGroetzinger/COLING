import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from progress.bar import Bar
import re
import argparse


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


def get_matching_sent_context(context, sent, windows=[1, 2, 3, 4, 5], use_sent_from_context=False):
    """
        Use this function to get closest match to a source_line or target_line in a paragraph.
    """
    sentence_tokenized_document = sentence_splitter(context)
    bleu_scores = []
    sents = []
    for elem in sentence_tokenized_document:
        reference = [word_tokenize(elem)]
        score = sentence_bleu(reference, sent)
        bleu_scores.append(score)
        sents.append(elem)
    index_of_max_bleu = bleu_scores.index(max(bleu_scores))
    matched_sent = sents[index_of_max_bleu]

    # make context here
    previous_sentences = []
    next_sentences = []
    sent_indexes = [i for i in range(len(sents))]
    for window in windows:
        next_sent_pos = index_of_max_bleu+window
        if next_sent_pos in sent_indexes:
            next_sent = sents[index_of_max_bleu+window]
            next_sentences.append(next_sent)
        # repeat for previous_sentences
        previous_sent_pos = index_of_max_bleu-window
        if previous_sent_pos in sent_indexes:
            previous_sent = sents[index_of_max_bleu - window]
            previous_sentences.append(previous_sent)

    previous_sentences.reverse()
    if use_sent_from_context:
        context = previous_sentences + [matched_sent] + next_sentences
    else:
        context = previous_sentences + [' '.join(sent)] + next_sentences
    return context


def add_filtered_context(list_of_wikihow_instances):
    print(len(list_of_wikihow_instances))
    bar = Bar('Processing ... ', max=len(list_of_wikihow_instances))
    new_wikihow_instances = []
    for wikihow_instance in list_of_wikihow_instances:
        bar.next()
        source_tokenized = wikihow_instance['Source_Line']
        target_tokenized = wikihow_instance['Target_Line']
        source_context_filtered = get_matching_sent_context(
            wikihow_instance['Source_Context'], source_tokenized)
        wikihow_instance['Source_Context_5'] = source_context_filtered
        #  get new context for target
        target_context_filtered = get_matching_sent_context(
            wikihow_instance['Target_Context'], target_tokenized)
        wikihow_instance['Target_Context_5'] = target_context_filtered
        new_wikihow_instances.append(wikihow_instance)
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
    print("------------------------------------")
    print("examples: ")
    for elem in new_wikihow_instances[0:10]:
        print(elem['Source_Line'])
        print(elem['Source_Context_5'])
    print("----------------------------------")

    try:
        assert len(wikihow_instances) == len(new_wikihow_instances)
    except AssertionError:
        print("Length is not the same: ")
        print("Length file-in: ", len(wikihow_instances))
        print("Length file-out: ", len(new_wikihow_instances))

    with open(filename_to_write, 'w') as json_out:
        json.dump(new_wikihow_instances, json_out)
