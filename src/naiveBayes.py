from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import preprocessing
import logging
import pickle
import os


def naive_bayes_2_points(n_gram=1, use_bern=False):
    if use_bern:
        file_name = "bern"
    else:
        file_name = "multi"
    model_file_name = 'naive_bayes_' + n_gram + '_gram_' + file_name + '2_points' + '.pkl'

    if os.path.isfile(model_file_name):
        logging.info("Found existing model trained with file: %s" % model_file_name)
        with open(model_file_name, 'rb') as model_file:
            clf = pickle.load(model_file)
    else:
        if use_bern:
            clf = BernoulliNB()
        else:
            clf = MultinomialNB()

        x_train, y_train, x_test, y_test = preprocessing.get_training_and_testing_for_2_points(n_gram=n_gram,
                                                                                               use_bern=use_bern)
        clf.fit(x_train, y_train)
        with open(model_file_name, 'wb') as output:
            pickle.dump(clf, output)

    score = clf.score(x_test, y_test)
    logging.info("test accuracy %.4f with %s gram %s trained by naive bayes for 2 points" % (score, n_gram, file_name))
    return clf


def naive_bayes_3_points(n_gram=1, use_bern=False):
    if use_bern:
        file_name = "bern"
    else:
        file_name = "multi"
    model_file_name = 'naive_bayes_' + n_gram + '_gram_' + file_name + '3_points' + '.pkl'

    if os.path.isfile(model_file_name):
        logging.info("Found existing model trained with file: %s" % model_file_name)
        with open(model_file_name, 'rb') as model_file:
            clf = pickle.load(model_file)
    else:
        if use_bern:
            clf = BernoulliNB()
        else:
            clf = MultinomialNB()

        x_train, y_train, x_test, y_test = preprocessing.get_training_and_testing_for_3_points(n_gram=n_gram,
                                                                                               use_bern=use_bern)
        clf.fit(x_train, y_train)
        with open(model_file_name, 'wb') as output:
            pickle.dump(clf, output)

    score = clf.score(x_test, y_test)
    logging.info("test accuracy %.4f with %s gram %s trained by naive bayes for 3 points" % (score, n_gram, file_name))
    return clf

if __name__ == '__main__':
    logging.basicConfig(filename='naive.log', level=logging.INFO, format='%(asctime)s %(message)s')
    naive_bayes_3_points(n_gram=1, use_bern=True)
    naive_bayes_3_points(n_gram=2, use_bern=True)
    naive_bayes_3_points(n_gram=3, use_bern=True)
    naive_bayes_3_points(n_gram=1)
    naive_bayes_3_points(n_gram=2)
    naive_bayes_3_points(n_gram=3)
    naive_bayes_2_points(n_gram=1)
