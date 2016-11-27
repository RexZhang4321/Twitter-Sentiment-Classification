import numpy as np
import pandas as pd
import re


hash_regex = re.compile(r"#(\w+)")
hndl_regex = re.compile(r"@(\w+)")
url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")


def load_data(path, names=None, usecols=None):
    data = pd.read_csv(path, header=None, names=names, usecols=usecols)
    data['class'] = data['class'].map({0: -1, 2: 0, 4: 1})
    return data.reindex(np.random.permutation(data.index))


def purify_row(row):
    row = re.sub(hash_regex, "", row)
    row = re.sub(hndl_regex, "", row)
    row = re.sub(url_regex, "", row)
    return row


def preprocess_data(data):
    vocab = {}
    new_data = []
    ct = 0
    for line in data:
        line = purify_row(line)
        new_data.append(line)
        chars = list(line)
        for c in chars:
            if c not in vocab:
                vocab[c] = ct
                ct += 1
    print vocab
    for i, line in enumerate(new_data):
        ints = map(lambda c: vocab[c], line)
        ints_str = ' '.join([str(c) for c in ints])
        new_data[i] = ints_str
    return new_data, vocab


def load_from_file(path, names=None, usecols=None):
    dt = load_data(path, names=names, usecols=usecols)
    y = dt['class'].values
    x = dt['data'].values
    x, vocab = preprocess_data(x)
    return x, y, vocab

if __name__ == '__main__':
    path = '../data/test.csv'
    names = ["class", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    x, y, vocab = load_from_file(path, names=names, usecols=usecols)
    print x
    print y
