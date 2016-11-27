import numpy as np
import pandas as pd
import re
import string
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
punctuation = string.punctuation


def load_data(path, names=None, usecols=None):
    data = pd.read_csv(path, header=None, names=names, usecols=usecols)
    data['class'] = data['class'].map({0: -1, 2: 0, 4: 1})
    return data.reindex(np.random.permutation(data.index))


def parse_row(row):
    _row = []
    row = re.sub(hash_regex, hash_repl, row)
    row = re.sub(hndl_regex, hndl_repl, row)
    row = re.sub(url_regex, "", row)
    for (repl, regx) in emoticons_regex:
        row = re.sub(regx, '', row)
    row = row.replace('\'', '')
    row = re.sub(word_bound_regex, punctuations_repl, row)
    row = re.sub(rpt_regex, rpt_repl, row)
    try:
        for word in word_tokenize(row):
            if word in stops or len(word) == 1:
                continue
            else:
                word = word.lower()
                _row.append(word)
    except:
        pass
    return _row


# Hashtags
hash_regex = re.compile(r"#(\w+)")


def hash_repl(match):
    return ''
    # return '__HASH_'+match.group(1).upper()


# Handels
hndl_regex = re.compile(r"@(\w+)")


def hndl_repl(match):
    return ''
    # return '__HNDL'#_'+match.group(1).upper()


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

# Punctuations
punctuations = [
    # ('',   ['.', ] ),\
    # ('',		[',', ] )	,\
    # ('',		['\'', '\"', ] )	,\
    ('__PUNC_EXCL', ['!', ]), \
    ('__PUNC_QUES', ['?', ]), \
    ('__PUNC_ELLP', ['...', ]), \
    ]


# For emoticon regexes
def escape_paren(arr):
    return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]


def regex_union(arr):
    return '(' + '|'.join(arr) + ')'


emoticons_regex = [(repl, re.compile(regex_union(escape_paren(regx))))
                   for (repl, regx) in emoticons]


# For punctuation replacement
def punctuations_repl(match):
    text = match.group(0)
    repl = []
    for (key, parr) in punctuations:
        for punc in parr:
            if punc in text:
                repl.append(key)
    # if (len(repl) > 0):
    #     return ' ' + ' '.join(repl) + ' '
    # else:
    return ' '


def convert2vec(text, maxLen):
    model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    data = np.zeros((len(dt), maxLen, 300))
    rowIndex = 0
    for row in text:
        wordIndex = 0
        for word in parse_row(row):
            if word in model:
                data[rowIndex, wordIndex, :] = model[word]
            else:
                data[rowIndex, wordIndex, :] = np.random.rand(300)
            wordIndex += 1
        rowIndex += 1
    return data


if __name__ == '__main__':
    path = 'C:/Users/zhang/Downloads/trainingandtestdata/training.csv'
    names = ["class", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    dt = load_data(path, names=names, usecols=usecols)
    dt = dt[:100000]
    text = dt['data'].as_matrix()
    labels = dt['class'].as_matrix()
    maxLen = 0
    for row in text:
        maxLen = max(len(parse_row(row)), maxLen)
    print 'Max length of words: %d' % maxLen
    data = convert2vec(text, maxLen)
    np.save('data.dat', data)
