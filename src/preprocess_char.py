import numpy as np
import pandas as pd
import re


hash_regex = re.compile(r"#")
hndl_regex = re.compile(r"@(\w+)")
url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")


def load_data(path, names=None, usecols=None):
    # data = pd.read_csv(path, delimiter="\t", header=None, names=names, usecols=usecols)
    data = pd.read_csv(path, header=None, names=names, usecols=usecols)
    # data['class'] = data['class'].map({0: 0, 2: 0, 4: 1})
    # data = data[~data['classes'].str.contains('neutral')]
    # data = data.replace({'classes': {'negative': 0, 'positive': 1}})
    data['classes'] = data['classes'].map({0: 0, 4: 1})
    return data.reindex(np.random.permutation(data.index))


def purify_row(row):
    row = re.sub(hash_regex, "", row)
    row = re.sub(hndl_regex, "", row)
    row = re.sub(url_regex, "", row)
    # remove non-ascii
    row = re.sub(r'[^\x00-\x7F]+', '', row)
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
    for i, line in enumerate(new_data):
        ints = map(lambda c: vocab[c], line)
        new_data[i] = ints
    return new_data, vocab


def load_from_file(path, names=None, usecols=None):
    dt = load_data(path, names=names, usecols=usecols)
    print dt
    y = dt['classes'].values
    y = np.array(y, dtype=np.int64)
    x = dt['data'].values
    x, vocab = preprocess_data(x)
    x = pad_mask(x)
    return x, y, vocab


def load_from_one_text(txt, vocab):
    txt = purify_row(txt)
    chars = list(txt)
    new_txt = []
    for c in chars:
        if c in vocab:
            new_txt.append(vocab[c])
    return pad_mask([new_txt])


def pad_mask(X, maxlen=140):
    N = len(X)
    X_out = np.zeros((N, maxlen, 2), dtype=np.int32)
    for i, x in enumerate(X):
        n = len(x)
        if n < maxlen:
            X_out[i, :n, 0] = x
            X_out[i, :n, 1] = 1
        else:
            X_out[i, :, 0] = x[:maxlen]
            X_out[i, :, 1] = 1
    return X_out

if __name__ == '__main__':
    path = '../data/training2.csv'
    names = ["classes", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    # path = '../data/semeval/train.tsv'
    # names = ["id", "classes", "data"]
    # usecols = [1, 2]
    x, y, vocab = load_from_file(path, names=names, usecols=usecols)
    # print y
    row = "~!@#$%^&*()_+`1234567890-=qwertyuiop[]asdfghjkl;'zxcvbnm,./QWERTYUIOP{}|ASDFGHJKL:ZXCVBNM<>?"
    print len(row)
    row = load_from_one_text(row, vocab)
    # print row
