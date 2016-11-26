import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
punctuation = string.punctuation
stemmer = nltk.stem.porter.PorterStemmer()


def load_data(path, names=None, usecols=None):
    data = pd.read_csv(path, header=None, names=names, usecols=usecols)
    data['class'] = data['class'].map({0: -1, 2: 0, 4: 1})
    return data.reindex(np.random.permutation(data.index))


def parse_row(row):
    _row = []
    row = re.sub(hash_regex, hash_repl, row)
    row = re.sub(hndl_regex, hndl_repl, row)
    row = re.sub(url_regex, "__URL", row)
    for (repl, regx) in emoticons_regex:
        row = re.sub(regx, ' ' + repl + ' ', row)
    row = row.replace('\'', '')
    row = re.sub(word_bound_regex, punctuations_repl, row)
    row = re.sub(rpt_regex, rpt_repl, row)
    try:
        for word in word_tokenize(row):
            # not pretty sure whether we need stopwords
            word = word.lower()
            word = stemmer.stem(word)
            _row.append(word)
            # if word in stops or len(word) == 1:
            #         continue
            # else:
            #     word = word.lower()
            #     word = stemmer.stem(word)
            #     _row.append(word)
    except:
        pass
    return _row


# Hashtags
hash_regex = re.compile(r"#(\w+)")
def hash_repl(match):
    return '__HASH_'+match.group(1).upper()


# Handels
hndl_regex = re.compile(r"@(\w+)")
def hndl_repl(match):
    return '__HNDL'#_'+match.group(1).upper()


# URLs
url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")


# Spliting by word boundaries
word_bound_regex = re.compile(r"\W+")


# Repeating words like hurrrryyyyyy
rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE);
def rpt_repl(match):
    return match.group(1)+match.group(1)


# Emoticons
emoticons = [
        ('__EMOT_SMILEY',   [':-)', ':)', '(:', '(-:', ]),
        ('__EMOT_LAUGH',    [':-D', ':D', 'X-D', 'XD', 'xD', ]),
        ('__EMOT_LOVE',     ['<3', ':\*', ]),
        ('__EMOT_WINK',		[';-)', ';)', ';-D', ';D', '(;', '(-;', ]),
        ('__EMOT_FROWN',    [':-(', ':(', '(:', '(-:', ]),
        ('__EMOT_CRY',      [':,(', ':\'(', ':"(', ':((']),
    ]


# Punctuations
punctuations = [
        #('',   ['.', ] ),\
        #('',		[',', ] )	,\
        #('',		['\'', '\"', ] )	,\
        ('__PUNC_EXCL',     ['!', ]),\
        ('__PUNC_QUES',     ['?', ]),\
        ('__PUNC_ELLP',     ['...', ]),\
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
    if (len(repl) > 0):
        return ' ' + ' '.join(repl) + ' '
    else:
        return ' '


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
    path = '../data/test.csv'
    names = ["class", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    dt = load_data(path, names=names, usecols=usecols)
    print dt
    for row in dt['data']:
        print parse_row(row)
