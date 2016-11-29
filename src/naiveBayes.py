from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import preprocessing
import logging
import pickle
import os


def naive_bayes_2_points(n_gram=1, use_bern=False):
    if use_bern:
        model_type = "bern"
    else:
        model_type = "multi"
    model_file_name = 'naive_bayes_' + str(n_gram) + '_gram_' + model_type + '_2_points' + '.pkl'
    model_file_path = os.path.join("../model", model_file_name)
    x_train, y_train, x_test, y_test = preprocessing.get_training_and_testing_for_2_points(n_gram=n_gram,
                                                                                           use_bern=use_bern)
    if os.path.isfile(model_file_path):
        logging.info("Found existing model trained with file: %s" % model_file_name)
        with open(model_file_path, 'rb') as model_file:
            clf = pickle.load(model_file)
    else:
        if use_bern:
            clf = BernoulliNB()
        else:
            clf = MultinomialNB()

        clf.fit(x_train, y_train)
        with open(model_file_path, 'wb') as output:
            pickle.dump(clf, output)

    score = clf.score(x_test, y_test)
    logging.info("test accuracy %.4f with %s gram %s trained by naive bayes for 2 points" % (score, str(n_gram),
                                                                                             model_type))
    return clf


def naive_bayes_3_points(n_gram=1, use_bern=False):
    if use_bern:
        model_type = "bern"
    else:
        model_type = "multi"
    model_file_name = 'naive_bayes_' + str(n_gram) + '_gram_' + model_type + '_3_points' + '.pkl'
    model_file_path = os.path.join("../model", model_file_name)
    x_train, y_train, x_test, y_test = preprocessing.get_training_and_testing_for_3_points(n_gram=n_gram,
                                                                                           use_bern=use_bern)
    if os.path.isfile(model_file_path):
        logging.info("Found existing model trained with file: %s" % model_file_name)
        with open(model_file_path, 'rb') as model_file:
            clf = pickle.load(model_file)
    else:
        if use_bern:
            clf = BernoulliNB()
        else:
            clf = MultinomialNB()

        clf.fit(x_train, y_train)
        with open(model_file_path, 'wb') as output:
            pickle.dump(clf, output)

    score = clf.score(x_test, y_test)
    logging.info("test accuracy %.4f with %s gram %s trained by naive bayes for 3 points" % (score, str(n_gram), model_type))
    return clf

if __name__ == '__main__':
    logging.basicConfig(filename='../running_log/naive.log', level=logging.INFO, format='%(asctime)s %(message)s')
    naive_bayes_3_points(n_gram=1, use_bern=True)
    naive_bayes_3_points(n_gram=2, use_bern=True)
    naive_bayes_3_points(n_gram=3, use_bern=True)
    naive_bayes_3_points(n_gram=1)
    naive_bayes_3_points(n_gram=2)
    naive_bayes_3_points(n_gram=3)
    naive_bayes_2_points(n_gram=1, use_bern=True)
    naive_bayes_2_points(n_gram=1)
