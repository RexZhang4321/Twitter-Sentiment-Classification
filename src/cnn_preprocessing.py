import numpy as np
import re
import string
import pandas as pd
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
model = gensim.models.Word2Vec.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
stops = set(stopwords.words("english"))
punctuation = string.punctuation
dataset = '2-points'
if dataset == '3-points':
    maxLen = 24
elif dataset == '2-points':
    maxLen = 30

def parse_row(row):
    _row = ' '
    # remove hashtags
    row = re.sub(hash_regex, '', row)
    # remove handles
    row = re.sub(hndl_regex, '', row)
    # remove urls
    row = re.sub(url_regex, '', row)
    # remove emoticons
    for (repl, regx) in emoticons_regex:
        row = re.sub(regx, '', row)
    # remove duplicate chars >= 3
    if dataset == '2-points':
        row = re.sub(word_bound_regex, ' ', row)
    row = re.sub(rpt_regex, rpt_repl, row)
    for word in word_tokenize(row):
        if word in stops or len(word) == 1:
            continue
        else:
            _row += ' '
            _row += word.lower()
    _row = re.sub(r"\'s", "", _row)
    _row = re.sub(r"[^A-Za-z0-9\'\s]", "", _row)
    _row = re.sub(r"\s{2,}", " ", _row)
    return _row.strip().lower()


# Hashtags
hash_regex = re.compile(r"#(\w+)")

# Handels
hndl_regex = re.compile(r"@(\w+)")

# URLs
url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")

# Spliting by word boundaries
word_bound_regex = re.compile(r"\W+")

# Repeating words like hurrrryyyyyy
rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)


def rpt_repl(match):
    return match.group(1) + match.group(1)


# Emoticons
emoticons = [
    ('__EMOT_SMILEY', [':-)', ':)', '(:', '(-:', ]),
    ('__EMOT_LAUGH', [':-D', ':D', 'X-D', 'XD', 'xD', ]),
    ('__EMOT_LOVE', ['<3', ':\*', ]),
    ('__EMOT_WINK', [';-)', ';)', ';-D', ';D', '(;', '(-;', ]),
    ('__EMOT_FROWN', [':-(', ':(', '(:', '(-:', ]),
    ('__EMOT_CRY', [':,(', ':\'(', ':"(', ':((']),
]


# For emoticon regexes
def escape_paren(arr):
    return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]


def regex_union(arr):
    return '(' + '|'.join(arr) + ')'


emoticons_regex = [(repl, re.compile(regex_union(escape_paren(regx))))
                   for (repl, regx) in emoticons]


def load_csv_data(path):
    names = ["class", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    data = pd.read_csv(path, header=None, names=names, usecols=usecols)
    data['class'] = data['class'].map({0: 0, 4: 1})
    data = data.reindex(np.random.permutation(data.index))
    x = data['data'].as_matrix()
    y = data['class'].as_matrix()
    return x, y


def load_tsv_data(path):
    with open(path, 'r') as tsv:
        data = [line.strip().split('\t') for line in tsv]
    data = np.array(data)
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    data = data[shuffle_indices]
    x = data[:, 2]
    _y = data[:, 1]
    y = np.zeros(len(_y))
    y[np.where(_y == 'positive')] = 2
    y[np.where(_y == 'neutral')] = 1
    y[np.where(_y == 'negative')] = 0
    return x, y.astype(int)


def load_data_and_labels(path):
    # Load data from files
    if dataset == '3-points':
        x_text, _y = load_tsv_data(path)
    elif dataset == '2-points':
        x_text, _y = load_csv_data(path)
    label_num = _y.max() + 1
    x_text = [parse_row(sent) for sent in x_text]
    # one hot
    y = np.zeros((len(_y), label_num))
    y[np.arange(len(_y)), _y] = 1
    # apply word2vec
    if dataset == '3-points':
        x_text = convert2vec(x_text, maxLen, model)
    return x_text, y


def convert2vec(text, maxLen, model):
    data = np.zeros((len(text), maxLen, 300))
    rowIndex = 0
    for row in text:
        wordIndex = 0
        for word in row.split(' '):
            if word in model:
                data[rowIndex, wordIndex, :] = model[word]
            else:
                data[rowIndex, wordIndex, :] = np.random.rand(300) * 0.5 - 0.25
            wordIndex += 1
            if wordIndex == maxLen:
                break
        rowIndex += 1
    return data


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data_x = data[0]
    data_y = data[1]
    data_size = len(data_y)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_x = data_x[shuffle_indices]
            shuffled_y = data_y[shuffle_indices]
        else:
            shuffled_x = data_x
            shuffled_y = data_y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield (shuffled_x[start_index:end_index], shuffled_y[start_index:end_index])

#
if __name__ == '__main__':
    x, y = load_data_and_labels('../data/semeval/test.tsv')
    # x, y = load_data_and_labels('../data/semeval/dev.tsv')
#     # maxLen = 0
#     # for row in x:
#     #     maxLen = max(len(row.split(' ')), maxLen)
#     #     if len(row.split(' ')) > 20:
#     #         print len(row.split(' '))
#     # print maxLen
