import argparse
import numpy as np
import os
import sys
import pandas as pd
import re

import warnings
warnings.filterwarnings('ignore')

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from conf.configure import Configure as conf

from utils import data_util
from utils.nltk_utils import tokenize_sentences
from utils.clean_util import TextCleaner
from utils.embedding_utils import read_embedding_list, clear_embedding_list, convert_tokens_to_ids
from utils.data_util import max_len

from utils.extra_utils import load_data, Embeds, Logger
from nltk.tokenize import RegexpTokenizer
from preprocessing import clean_text, convert_text2seq, get_embedding_matrix, split_data

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"
sentences_length = max_len()
remove_stop_words = True
stem_words = False #True
swear_words_fname = '../input/swear_words.csv'
wrong_words_fname = '../input/correct_words.csv'

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def main():
    
    print("Loading data...")
    train_data = pd.read_csv(conf.train_data_path)
    test_data = pd.read_csv(conf.x_test_data_path)
    
    print("Cleaning text...")
    train_data["comment_text"] = train_data["comment_text"].apply(TextCleaner.clean_text2)
    test_data["comment_text"] = test_data["comment_text"].apply(TextCleaner.clean_text2)
    print("remove_stop_words ", remove_stop_words)
    print("stem_words", stem_words)
    train_data["comment_text"] = train_data["comment_text"].apply(lambda x: TextCleaner.clean_text(x, remove_stop_words=remove_stop_words, stem_words=stem_words))
    test_data["comment_text"] = test_data["comment_text"].apply(lambda x: TextCleaner.clean_text(x, remove_stop_words=remove_stop_words, stem_words=stem_words))

#     extra preprocessing
    print('extra cleaning...')
    swear_words = load_data(swear_words_fname, func=lambda x: set(x.T[0]), header=None)
    wrong_words_dict = load_data(wrong_words_fname, func=lambda x: {val[0] : val[1] for val in x})
    tokinizer = RegexpTokenizer(r'\w+')
    regexps = [re.compile("([a-zA-Z]+)([0-9]+)"), re.compile("([0-9]+)([a-zA-Z]+)")]
    train_data["comment_text"] = clean_text(train_data["comment_text"], tokinizer, wrong_words_dict, regexps)
    train_data["comment_text"] = clean_text(train_data["comment_text"], tokinizer, wrong_words_dict, regexps)

    list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
    list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values
    y_train = train_data[CLASSES].values

    print("Tokenizing sentences in train set...")
    tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})

    print("Tokenizing sentences in test set...")
    tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)

    words_dict[UNKNOWN_WORD] = len(words_dict)

    print("Loading embeddings...")
    embedding_list, embedding_word_dict = read_embedding_list(conf.embedding_path)
    embedding_size = len(embedding_list[0])

    print("Preparing data...")
    embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append([0.] * embedding_size)
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append([-1.] * embedding_size)

    embedding_matrix = np.array(embedding_list)
    print("saving embedding matrix")
    data_util.save_embedding_matrix(embedding_matrix)

    id_to_word = dict((id, word) for word, id in words_dict.items())
    train_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_train,
        id_to_word,
        embedding_word_dict,
        sentences_length)
    test_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_test,
        id_to_word,
        embedding_word_dict,
        sentences_length)
    x_train = np.array(train_list_of_token_ids)
    x_test = np.array(test_list_of_token_ids)
    
    print('x_train:', x_train.shape, ', y_train:', y_train.shape, ', x_test.shape', x_test.shape)
    print("Save train and test data...")
    data_util.save_processed_dataset(x_train, y_train, x_test)

if __name__ == "__main__":
    main()
