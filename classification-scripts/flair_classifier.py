from flair.embeddings import BertEmbeddings, Sentence, DocumentPoolEmbeddings, DocumentLSTMEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.data_fetcher import NLPTaskDataFetcher
from flair.trainers import ModelTrainer
from pathlib import Path


# load corpus
corpus = NLPTaskDataFetcher.load_classification_corpus(
    Path('./'), test_file='test.tsv', dev_file='dev.tsv', train_file='train.tsv')

# load embeddings
bert_embedding = BertEmbeddings()
document_embeddings = DocumentRNNEmbeddings(
    [bert_embedding], reproject_words_dimension=512, rnn_type='LSTM')

# load classifier
classifier = TextClassifier(
    document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)

trainer = ModelTrainer(classifier, corpus)

trainer.train('./', max_epochs=10)
