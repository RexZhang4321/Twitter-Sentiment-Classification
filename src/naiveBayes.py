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

    if os.path.isfile(model_file_path):
        logging.info("Found existing model trained with file: %s" % model_file_name)
        with open(model_file_path, 'rb') as model_file:
            model = pickle.load(model_file)
            clf = model["clf"]
            dic = model["dic"]
            x_test, y_test = preprocessing.get_testing_for_2_points(dic, n_gram=n_gram, use_bern=use_bern)
    else:
        if use_bern:
            clf = BernoulliNB()
        else:
            clf = MultinomialNB()
        x_train, y_train, x_test, y_test, dic = preprocessing.get_training_and_testing_for_2_points(n_gram=n_gram,
                                                                                                    use_bern=use_bern)
        clf.fit(x_train, y_train)
        with open(model_file_path, 'wb') as output:
            model = dict()
            model["clf"] = clf
            model["dic"] = dic
            pickle.dump(model, output)

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

    if os.path.isfile(model_file_path):
        logging.info("Found existing model trained with file: %s" % model_file_name)
        with open(model_file_path, 'rb') as model_file:
            model = pickle.load(model_file)
            clf = model["clf"]
            dic = model["dic"]
            x_test, y_test = preprocessing.get_testing_for_3_points(dic, n_gram=n_gram, use_bern=use_bern)
    else:
        if use_bern:
            clf = BernoulliNB()
        else:
            clf = MultinomialNB()
        x_train, y_train, x_test, y_test, dic = preprocessing.get_training_and_testing_for_3_points(n_gram=n_gram,
                                                                                                    use_bern=use_bern)
        clf.fit(x_train, y_train)
        with open(model_file_path, 'wb') as output:
            model = dict()
            model["clf"] = clf
            model["dic"] = dic
            pickle.dump(model, output)

    score = clf.score(x_test, y_test)
    logging.info("test accuracy %.4f with %s gram %s trained by naive bayes for 3 points" % (score, str(n_gram), model_type))
    return clf


class NBC():

    def __init__(self, n_points=2, n_gram=1, use_bern=False):
        self.n_points = n_points
        self.n_gram = n_gram
        self.use_bern = use_bern
        if use_bern:
            model_type = "bern"
        else:
            model_type = "multi"
        model_file_name = 'naive_bayes_' + str(n_gram) + '_gram_' + model_type + '_' + str(n_points) + '_points' + '.pkl'
        print model_file_name
        model_file_path = os.path.join("../model", model_file_name)
        if os.path.isfile(model_file_path):
            with open(model_file_path, 'rb') as model_file:
                model = pickle.load(model_file)
                self.clf = model["clf"]
                self.dic = model["dic"]
        else:
            print "no corresponding model found! exiting..."
            exit(1)

    def predict(self, txt):
        txt = preprocessing.generate_BOW(txt, self.dic, n_gram=self.n_gram, use_bern=self.use_bern)
        y_pred = self.clf.predict(txt)
        return y_pred


if __name__ == '__main__':
    '''
    logging.basicConfig(filename='../running_log/naive.log', level=logging.INFO, format='%(asctime)s %(message)s')
    naive_bayes_3_points(n_gram=1, use_bern=True)
    naive_bayes_3_points(n_gram=2, use_bern=True)
    naive_bayes_3_points(n_gram=3, use_bern=True)
    naive_bayes_3_points(n_gram=1)
    naive_bayes_3_points(n_gram=2)
    naive_bayes_3_points(n_gram=3)
    naive_bayes_2_points(n_gram=1, use_bern=True)
    naive_bayes_2_points(n_gram=1)
    '''
    nbc = NBC(3, n_gram=3, use_bern=False)
    print nbc.predict(["hi", "good", "bad"])
