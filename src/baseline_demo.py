import preprocessing
import naiveBayes
import svm
import pickle

with open('../model/naive_bayes_3_gram_multi_3_points.pkl', 'r') as fp:
    naive_3 = pickle.load(fp)
    naive_3_clf = naive_3["clf"]
    naive_3_dict = naive_3["dic"]

with open('../model/naive_bayes_1_gram_multi_2_points.pkl', 'r') as fp:
    naive_2 = pickle.load(fp)
    naive_2_clf = naive_2["clf"]
    naive_2_dict = naive_2["dic"]

with open('../model/svm_3_gram_3_points.pkl', 'r') as fp:
    svm_3 = pickle.load(fp)
    svm_3_clf = svm_3["clf"]
    svm_3_dict = svm_3["dic"]

with open('../model/svm_1_gram_2_points.pkl', 'r') as fp:
    svm_2 = pickle.load(fp)
    svm_2_clf = svm_2["clf"]
    svm_2_dict = svm_2["dic"]


def naive_bayes_predict_2_points(rows):
    data = preprocessing.generate_BOW(rows, naive_2_dict)
    y_predict = naive_2_clf.predict(data)
    return list(y_predict)


def naive_bayes_predict_3_points(rows):
    data = preprocessing.generate_BOW(rows, naive_3_dict)
    y_predict = naive_3_clf.predict(data)
    return list(y_predict)


def svm_predict_2_points(rows):
    data = preprocessing.generate_BOW(rows, svm_2_dict)
    y_predict = svm_2_clf.predict(data)
    return list(y_predict)


def svm_predict_3_points(rows):
    data = preprocessing.generate_BOW(rows, svm_3_dict)
    y_predict = svm_3_clf.predict(data)
    return list(y_predict)

if __name__ == '__main__':
    rows = ["I love America", "I hate America", "Test"]
    print(naive_bayes_predict_2_points(rows))
    print(naive_bayes_predict_3_points(rows))
    print(svm_predict_2_points(rows))
    print(svm_predict_3_points(rows))

