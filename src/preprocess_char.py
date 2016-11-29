import numpy as np
import pandas as pd
import re


hash_regex = re.compile(r"#")
hndl_regex = re.compile(r"@(\w+)")
url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")


def load_data(path, names=None, usecols=None, mode='semeval2'):
    np.random.seed(1234)
    if mode == 'semeval2':
        names = ["id", "classes", "data"]
        usecols = [1, 2]
        data = pd.read_csv(path, delimiter="\t", header=None, names=names, usecols=usecols)
        data = data[~data['classes'].str.contains('neutral')]
        data = data.replace({'classes': {'negative': 0, 'positive': 1}})
    elif mode == 'semeval3':
        names = ["id", "classes", "data"]
        usecols = [1, 2]
        data = pd.read_csv(path, delimiter="\t", header=None, names=names, usecols=usecols)
        data = data.replace({'classes': {'negative': 0, 'neutral': 1, 'positive': 2}})
    elif mode == 'senti':
        names = ["classes", "id", "time", "query", "user", "data"]
        usecols = [0, 5]
        data = pd.read_csv(path, header=None, names=names, usecols=usecols)
        data['classes'] = data['classes'].map({0: 0, 4: 1})
    else:
        print "unrecognized dataset. abort"
        exit(1)
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


def load_from_file(path, names=None, usecols=None, mode='semeval2'):
    dt = load_data(path, names=names, usecols=usecols, mode=mode)
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


def load_from_file_with_vocab(path, vocab, names=None, usecols=None, mode='semeval2'):
    dt = load_data(path, names=names, usecols=usecols, mode=mode)
    print dt
    y = dt['classes'].values
    x = dt['data'].values
    x = [load_from_one_text(row, vocab)[0] for row in x]
    return np.array(x), y


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
    x, y, vocab = load_from_file("../data/training2.csv", mode='senti')
    print x
    print y
    print len(vocab)
    print vocab
