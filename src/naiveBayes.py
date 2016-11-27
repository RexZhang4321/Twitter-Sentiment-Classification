from sklearn.naive_bayes import MultinomialNB
import preprocessing
import numpy as np


def naive_bayes_2_points():
    path = '../data/training.csv'
    names = ["class", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    dt = preprocessing.load_data(path, names=names, usecols=usecols)
    print "loading finished"
    n_records = 5000 # 74.47% in 50000 samples
    dic = preprocessing.generate_dict_for_BOW(dt[:n_records])
    print "dict size:", len(dic)
    x_train, y_train, x_test, y_test = preprocessing.get_training_and_testing(dt[:n_records])
    x_train = preprocessing.generate_BOW(x_train, dic)
    x_test = preprocessing.generate_BOW(x_test, dic)
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    print "training finished"
    print clf.score(x_test, y_test)


# hand written 3 point NBC classifier
# for each record, there are two probabilities, one for positive, one for negatve
# we define, if one probability is larger enough, we will classify this record to that sentiment
# otherwise (two probabilities are close), we define it neutural
class MNBC:

    def __init__(self, diff=0):
        self.diff = diff

    def set_diff(self, diff):
        self.diff = diff

    def fit(self, x_train, y_train):
        n_train = len(x_train)
        n_word = len(x_train[0])
        thetaPos = [0] * n_word
        thetaNeg = [0] * n_word
        for i in range(0, n_train):
            cur = thetaPos if y_train[i] == 1 else thetaNeg
            for j in range(0, n_word):
                cur[j] += x_train[i][j]
        total_pos_words = sum(thetaPos)
        total_neg_words = sum(thetaNeg)
        thetaPos = (np.array(thetaPos) + 1.0) / (total_pos_words + n_word)
        thetaNeg = (np.array(thetaNeg) + 1.0) / (total_neg_words + n_word)
        self.thetaPos = np.log(thetaPos)
        self.thetaNeg = np.log(thetaNeg)

    def predict(self, x_test):
        x_test = np.array(x_test)
        pos_vec = x_test.dot(self.thetaPos.T)
        neg_vec = x_test.dot(self.thetaNeg.T)
        y_pred = []
        # debug
        diffs = []
        for i in range(0, len(pos_vec)):
            diffs.append(abs(pos_vec[i] - neg_vec[i]))
            # neutral
            if abs(pos_vec[i] - neg_vec[i]) <= self.diff:
                y_pred.append(0)
            # positive
            elif pos_vec[i] > neg_vec[i]:
                y_pred.append(1)
            # negative
            else:
                y_pred.append(-1)
        with open('diffs', 'w') as f:
            f.write(str(diffs))
        return y_pred

    def accuracy(self, y_pred, y_true):
        hit = 0.0
        ttl = len(y_pred)
        assert(len(y_pred) == len(y_true))
        for i in range(0, ttl):
            if y_pred[i] == y_true[i]:
                hit += 1
        return hit / ttl

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return self.accuracy(y_pred, y_test)


def naive_bayes_3_points():
    path = '../data/training2.csv'
    names = ["class", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    dt = preprocessing.load_data(path, names=names, usecols=usecols)
    print "loading finished"
    n_records = 10000
    dic = preprocessing.generate_dict_for_BOW(dt[:n_records])
    print "dict size:", len(dic)
    words_train = dt['data'][:n_records].values
    y_train = dt['class'][:n_records].values
    dt2 = preprocessing.load_data('../data/testing.csv', names=names, usecols=usecols)
    words_test = dt2['data'].values
    y_test = dt2['class'].values
    # print len(x_test), len(y_test)
    x_train = preprocessing.generate_BOW(words_train, dic)
    x_test = preprocessing.generate_BOW(words_test, dic)
    clf = MNBC(diff=0)
    for i in range(0, 100):
        diff = 0.0 + i / 100.0
        clf.set_diff(diff=diff)
        clf.fit(x_train, y_train)
        print diff, clf.score(x_test, y_test)

if __name__ == '__main__':
    #naive_bayes_2_points()
    naive_bayes_3_points()
