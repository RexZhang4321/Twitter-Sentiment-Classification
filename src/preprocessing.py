import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
punctuation = string.punctuation
ascii = string.ascii_lowercase
stemmer = nltk.stem.porter.PorterStemmer()


def load_data(path, names=None, usecols=None):
    data = pd.read_csv(path, header=None, names=names, usecols=usecols)
    return data.reindex(np.random.permutation(data.index))


def parse_row(row):
    _row = []
    try:
        for word in word_tokenize(row):
            if word in stops or len(word) == 1:
                    continue
            else:
                word = word.lower()
                word = stemmer.stem(word)
                _row.append(word)
    except:
        pass
    return _row


def generate_dict_for_BOW(dt):
    word_dic = {}
    for row in dt["data"]:
        for word in parse_row(row):
            if word in word_dic:
                word_dic[word] += 1
            else:
                word_dic[word] = 1
    threshold = 1
    unk = 0
    for key, value in word_dic.items():
        if value <= threshold:
            del word_dic[key]
            unk += 1
    word_dic['UNK'] = unk
    return word_dic.keys()


def generate_BOW(dt, keys):
    data = []
    for row in dt:
        words = np.zeros(len(keys))
        for word in parse_row(row):
            if word in keys:
                words[keys.index(word)] += 1
            else:
                words[keys.index('UNK')] += 1
        data.append(words)
    return data


def get_training_and_testing(data):
    n_train = int(len(data) * 0.8)
    x_train = data["data"][:n_train]
    y_train = data["class"][:n_train]
    print pd.DataFrame(y_train).groupby("class").size()
    x_test = data["data"][n_train:]
    y_test = data["class"][n_train:]
    print pd.DataFrame(y_test).groupby("class").size()
    return x_train.values, y_train.values, x_test.values, y_test.values

if __name__ == '__main__':
    path = '../data/training.csv'
    names = ["class", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    dt = load_data(path, names=names, usecols=usecols)
    print "loading finished"
    # get training set and test set in numpy representation
    dic = generate_dict_for_BOW(dt[:100])
    print "generate dic finished"
    print dic
    x_train, y_train, x_test, y_test = get_training_and_testing(dt[:100])
    x_train = generate_BOW(x_train, dic)
    x_test = generate_BOW(x_test, dic)
    print len(x_train)
    print len(x_test)
