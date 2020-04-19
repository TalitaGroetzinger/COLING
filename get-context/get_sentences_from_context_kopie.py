import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from progress.bar import Bar
import re
import argparse
import pdb
from pprint import pprint


def remove_left_timestamps(left_context):
    for sent in left_context:
        if '## Timestamp' in sent:
            timestamp_index = left_context.index(sent)
            if len(sent.split(" ")) <= 2:
                left_context = left_context[timestamp_index+1:]
            else:
                context = left_context[timestamp_index+1:]
                # check de timestamp in het element zelf
                timestamp_sent = left_context[timestamp_index].split()
                for word in timestamp_sent:
                    if "Timestamp" in word:
                        timestamp_index_within_sent = timestamp_sent.index(
                            word)
                        content_from_timestamp = ' '.join(
                            timestamp_sent[timestamp_index_within_sent+1:])
                        left_context = content_from_timestamp + \
                            ' ' + ' '.join(context)
                        return left_context

    return left_context


def remove_right_timestamps(right_context):
    for sent in right_context:
        if '## Timestamp' in sent:
            try:
                timestamp_index = right_context.index(sent)
            except ValueError:
                print("=================")
                print(sent)
                print(right_context)
                print("=================")
            if len(sent.split(" ")) <= 2:
                right_context = right_context[timestamp_index-1]
            else:
                timestamp_index = right_context.index(sent)
                # pak in ieder geval alles voor de timestamp
                context = right_context[:timestamp_index]
                timestamp_sent = right_context[timestamp_index].split()

                for word in timestamp_sent:
                    if "Timestamp" in word:
                        timestamp_index_within_sent = timestamp_sent.index(
                            word)
                        timestamp_content = timestamp_sent[:timestamp_index_within_sent-1]
                        right_context = ' '.join(
                            context) + ' ' + ' '.join(timestamp_content)
                        return right_context
    return right_context


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def sentence_splitter(document):
    """
      Sentence splitter to deal with bullet items in texts.
    """
    # Tokenize per 'sub sentence list' instead of joining (to keep markdown headers separated)
    if not document:
        return []
    sentences = iter(sent_tokenize(document))
    # Flatten sentences: (https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists)
    # sentences = (
    #    sentence for sub_sentences in unflattened_sentences for sentence in sub_sentences)

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
    try:
        splitted_left = sentence_splitter(remove_html_tags(context['left']))
        splitted_current = sentence_splitter(
            remove_html_tags(context['current']))
        splitted_right = sentence_splitter(remove_html_tags(context['right']))
    except StopIteration:
        pdb.set_trace()
    # find the Source_line or target line in the context provided in current.
    bleu_scores = []
    sents = []
    if tokenized:
        tokenized_sent = word_tokenize(sent)

    else:
        tokenized_sent = sent

    for elem in splitted_current:
        reference = [word_tokenize(elem)]
        score = sentence_bleu(reference, tokenized_sent)
        bleu_scores.append(score)
        sents.append(elem)
    index_of_max_bleu = bleu_scores.index(max(bleu_scores))
    matched_sent = sents[index_of_max_bleu]

    # current might have items to the left or right already.
    left_items = splitted_current[:index_of_max_bleu]
    sentences_left = left_items
    left_index = max(5 - len(sentences_left), 0)
    left_items = splitted_left[-1*left_index:] + left_items

    right_items = splitted_current[index_of_max_bleu+1:]

    sentences_right = right_items
    right_index = max(5 - len(sentences_right), 0)
    right_items.extend(splitted_right[:right_index])

    return {

        "left": remove_left_timestamps(left_items),
        "current": splitted_current[index_of_max_bleu],
        "right": remove_right_timestamps(right_items)
    }


def add_filtered_context(list_of_wikihow_instances):
    print(len(list_of_wikihow_instances))
    bar = Bar('Processing ... ', max=len(list_of_wikihow_instances))
    new_wikihow_instances = []
    for index, wikihow_instance in enumerate(list_of_wikihow_instances):
        bar.next()

        source_context_filtered = get_matching_sent_context(
            wikihow_instance['Source_Context_New'], wikihow_instance['Source_Line'])

        wikihow_instance['Source_Context_5'] = source_context_filtered

        target_context_filtered = get_matching_sent_context(
            wikihow_instance['Target_Context_New'], wikihow_instance['Target_Line'])
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

    try:
        assert len(wikihow_instances) == len(new_wikihow_instances)
    except AssertionError:
        print("Length is not the same: ")
        print("Length file-in: ", len(wikihow_instances))
        print("Length file-out: ", len(new_wikihow_instances))

    with open(filename_to_write, 'w') as json_out:
        json.dump(new_wikihow_instances, json_out)
