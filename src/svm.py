import preprocessing
from sklearn.svm import SVC
import numpy as np
import logging
import threading
from sklearn.metrics import accuracy_score
import os
import pickle


def svm_2_points(n_gram=1):
    model_file_name = 'svm_' + str(n_gram) + '_gram_' + '2_points' + '.pkl'
    model_file_path = os.path.join("../model", model_file_name)
    if os.path.isfile(model_file_path):
        logging.info("Found existing model trained with file: %s" % model_file_name)
        with open(model_file_path, 'rb') as model_file:
            best_clf = pickle.load(model_file)
    else:
        x_train, y_train, x_test, y_test = preprocessing.get_training_and_testing_for_2_points(n_gram=n_gram)
        best_clf, best_score = train_and_select_model(x_train, y_train)
        logging.info("best validation accuracy %.4f" % best_score)
        with open(model_file_path, 'wb') as output:
            pickle.dump(best_clf, output)

    score = best_clf.score(x_test, y_test)
    logging.info("test accuracy %.4f with %s gram trained by svm for 2 points" % (score, str(n_gram)))
    return best_clf


def svm_3_points(n_gram=1):
    model_file_name = 'svm_' + str(n_gram) + '_gram_' + '3_points' + '.pkl'
    model_file_path = os.path.join("../model", model_file_name)
    if os.path.isfile(model_file_path):
        logging.info("Found existing model trained with file: %s" % model_file_name)
        with open(model_file_path, 'rb') as model_file:
            best_clf = pickle.load(model_file)
    else:
        x_train, y_train, x_test, y_test = preprocessing.get_training_and_testing_for_3_points(n_gram=n_gram)
        best_clf, best_score = train_and_select_model(x_train, y_train)
        logging.info("best validation accuracy %.4f" % best_score)
        with open(model_file_path, 'wb') as output:
            pickle.dump(best_clf, output)

    score = best_clf.score(x_test, y_test)
    logging.info("test accuracy %.4f with %s gram trained by svm for 3 points" % (score, str(n_gram)))
    return best_clf


def cv_worker(param, x_train, y_train, clf_map):
    k_folder = 3
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    splitted_x = np.array_split(x_train, k_folder)
    splitted_y = np.array_split(y_train, k_folder)

    data_rows = len(y_train)
    data_columns_x = len(x_train[0])
    data_columns_y = 1

    train_score = 0
    validation_score = 0
    svc = SVC(**param)
    for i in range(0, k_folder):
        validation_x = splitted_x[i]
        validation_y = splitted_y[i]
        validation_rows = len(splitted_y[i])
        train_rows = data_rows - validation_rows

        train_x = np.zeros((train_rows, data_columns_x))
        train_y = np.zeros((train_rows, data_columns_y))

        current = 0
        for j in range(0, i):
            rows = len(splitted_y[j])
            for k in range(0, rows):
                train_x[current + k] = splitted_x[j][k]
                train_y[current + k] = splitted_y[j][k]
            current += rows

        for j in range(i + 1, k_folder):
            rows = len(splitted_y[j])
            for k in range(0, rows):
                train_x[current + k] = splitted_x[j][k]
                train_y[current + k] = splitted_y[j][k]
            current += rows
        train_y = train_y.flatten()
        logging.info("Learning Model under folder %s and params %s" % (i + 1, str(param.items())))
        svc.fit(train_x, train_y)
        logging.info("Running prediction under folder %s and params %s" % (i + 1, str(param.items())))
        train_predict_y = svc.predict(train_x)
        validation_predict_y = svc.predict(validation_x)
        train_score += accuracy_score(train_y, train_predict_y)
        validation_score += accuracy_score(validation_y, validation_predict_y)
    validation_score /= k_folder
    train_score /= k_folder
    clf_map[validation_score] = svc
    logging.info("train accuracy %.4f with params %s" % (train_score, str(param.items())))
    logging.info("validation accuracy %.4f with params %s" % (validation_score, str(param.items())))


def train_and_select_model(x_train, y_train):
    param_set = [
        #{'kernel': 'rbf', 'C': 5, 'gamma': 0.05},
        #{'kernel': 'rbf', 'C': 2, 'gamma': 0.05},
        #{'kernel': 'rbf', 'C': 2, 'gamma': 0.06},
        {'kernel': 'linear', 'C': 0.09},
        {'kernel': 'linear', 'C': 0.1},
    ]
    clf_map = dict()
    for param in param_set:
        t = threading.Thread(target=cv_worker, args=(param, x_train, y_train, clf_map,))
        t.start()

    logging.debug('Waiting for worker threads')
    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is not main_thread:
            t.join()

    best_score = max(clf_map.keys())
    best_clf = clf_map[best_score]

    return best_clf, best_score


if __name__ == '__main__':
    logging.basicConfig(filename='svm.log', level=logging.INFO, format='%(asctime)s %(message)s')
    svm_3_points(n_gram=1)
    svm_3_points(n_gram=2)
    svm_3_points(n_gram=3)
    svm_2_points(n_gram=1)

