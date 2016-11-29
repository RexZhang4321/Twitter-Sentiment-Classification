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


def load_data(path, class_map, sep=",", names=None, usecols=None):
    data = pd.read_csv(path, sep=sep, header=None, names=names, usecols=usecols)
    data['class'] = data['class'].map(class_map)
    return data.reindex(np.random.permutation(data.index))


def parse_row(row):
    _row = []
    row = re.sub(hash_regex, hash_repl, row)
    row = re.sub(user_regex, '__USERNAME', row)
    row = re.sub(url_regex, "__URL", row)
    for (repl, regx) in emoticons_regex:
        row = re.sub(regx, ' ' + repl + ' ', row)
    row = row.replace('\'', '')
    row = re.sub(word_bound_regex, punctuations_repl, row)
    row = re.sub(rpt_regex, rpt_repl, row)
    try:
        for word in word_tokenize(row):
            if word in stops or len(word) <= 2:
                continue
            else:
                if word[0:2] != '__':
                    word = word.lower()
                word = stemmer.stem(word)
                _row.append(word)
    except:
        pass
    return _row


def extract_bigram_after_parsing(row):
    return [bg for bg in nltk.bigrams(row)]


def extract_trigram_after_parsing(row):
    return [tg for tg in nltk.trigrams(row)]

# Hashtags
hash_regex = re.compile(r"#(\w+)")


def hash_repl(match):
    return '__HASH_' + match.group(1).upper()


# USERNAME
user_regex = re.compile(r"@(\w+)")

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
        ('__EMOT_SMILEY',   [':-)', ':)', '(:', '(-:', ]),
        ('__EMOT_LAUGH',    [':-D', ':D', 'X-D', 'XD', 'xD', ]),
        ('__EMOT_LOVE',     ['<3', ':\*', ]),
        ('__EMOT_WINK',		[';-)', ';)', ';-D', ';D', '(;', '(-;', ]),
        ('__EMOT_FROWN',    [':-(', ':(', '(:', '(-:', ]),
        ('__EMOT_CRY',      [':,(', ':\'(', ':"(', ':((']),
    ]


# Punctuations
punctuations = {'!': '__PUNC_EXCL', '?': '__PUNC_QUES', '...': '__PUNC_ELLP'}


# For emoticon regex
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
    for (key, val) in punctuations.items():
        if key in text:
            repl.append(val)
    if len(repl) > 0:
        return ' ' + ' '.join(repl) + ' '
    else:
        return ' '


def generate_dict_for_BOW(dt, n_gram=1):
    word_dic = {}
    for row in dt["data"]:
        parsed_row = parse_row(row)
        if n_gram >= 2:
            parsed_row_bi = extract_bigram_after_parsing(parsed_row)
        else:
            parsed_row_bi = []

        if n_gram >= 3:
            parsed_row_tri = extract_trigram_after_parsing(parsed_row)
        else:
            parsed_row_tri = []

        for word in parsed_row + parsed_row_bi + parsed_row_tri:
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


def generate_BOW(dt, keys, n_gram=1, use_bern=False):
    data = []
    for row in dt:
        words = np.zeros(len(keys))
        parsed_row = parse_row(row)

        if n_gram >= 2:
            parsed_row_bi = extract_bigram_after_parsing(parsed_row)
        else:
            parsed_row_bi = []

        if n_gram >= 3:
            parsed_row_tri = extract_trigram_after_parsing(parsed_row)
        else:
            parsed_row_tri = []

        for word in parsed_row + parsed_row_bi + parsed_row_tri:
            if word in keys:
                if use_bern:
                    words[keys.index(word)] = 1
                else:
                    words[keys.index(word)] += 1
            else:
                if use_bern:
                    words[keys.index('UNK')] = 1
                else:
                    words[keys.index('UNK')] += 1
        data.append(words)
    return data


def get_training_and_testing(data):
    n_train = int(len(data) * 0.99)
    x_train = data["data"][:n_train]
    y_train = data["class"][:n_train]
    print pd.DataFrame(y_train).groupby("class").size()
    x_test = data["data"][n_train:]
    y_test = data["class"][n_train:]
    print pd.DataFrame(y_test).groupby("class").size()
    return x_train.values, y_train.values, x_test.values, y_test.values


def get_data_and_label(data):
    n = len(data)
    x = data["data"][:n]
    y = data["class"][:n]
    return x, y


def get_training_and_testing_for_2_points(n_gram=1, use_bern=False):
    path = '../data/training.csv'
    names = ["id", "class", "data"]
    dt = load_data(path, {0: -1, 2: 0, 4: 1}, sep="\t", names=names)
    print "loading finished"
    n_records = 400000  # 74.47% in 50000 samples
    dic = generate_dict_for_BOW(dt[:n_records], n_gram=n_gram)
    print "dict size:", len(dic)
    x_train, y_train = get_data_and_label(dt[:n_records])
    x_train = generate_BOW(x_train, dic, n_gram=n_gram, use_bern=use_bern)

    test_path = '../data/testing.csv'
    test_dt = load_data(test_path, {0: -1, 2: 0, 4: 1}, sep="\t", names=names)
    x_test, y_test = get_data_and_label(test_dt)
    x_test = generate_BOW(x_test, dic, n_gram=n_gram, useBern=use_bern)
    return x_train, y_train, x_test, y_test


def get_training_and_testing_for_3_points(n_gram=1, use_bern=False):
    path = '../data/semeval/train.tsv'
    names = ["id", "class", "data"]
    dt = load_data(path, {"negative": -1, "neutral": 0, "positive": 1}, sep="\t", names=names)
    print "loading finished"
    dic = generate_dict_for_BOW(dt, n_gram=n_gram)
    print "dict size:", len(dic)
    x_train, y_train = get_data_and_label(dt)
    x_train = generate_BOW(x_train, dic, n_gram=n_gram, use_bern=use_bern)

    test_path = '../data/semeval/test.tsv'
    test_dt = load_data(test_path, {"negative": -1, "neutral": 0, "positive": 1}, sep="\t", names=names)
    x_test, y_test = get_data_and_label(test_dt)
    x_test = generate_BOW(x_test, dic, n_gram=n_gram, use_bern=use_bern)
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    path = '../data/test.csv'
    names = ["class", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    dt = load_data(path, names=names, usecols=usecols)
    print dt
    for row in dt['data']:
        print parse_row(row)
