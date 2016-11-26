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


def load_from_file():
    pass

if __name__ == '__main__':
    print purify_row("#sdc asd @sadv sdv http://asdn oh")
