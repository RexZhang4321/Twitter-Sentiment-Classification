from sklearn.naive_bayes import MultinomialNB
import preprocessing

if __name__ == '__main__':
    path = '../data/training.csv'
    names = ["class", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    dt = preprocessing.load_data(path, names=names, usecols=usecols)
    print "loading finished"
    n_records = 100000
    dic = preprocessing.generate_dict_for_BOW(dt[:n_records])
    print "dict size:", len(dic)
    x_train, y_train, x_test, y_test = preprocessing.get_training_and_testing(dt[:n_records])
    x_train = preprocessing.generate_BOW(x_train, dic)
    x_test = preprocessing.generate_BOW(x_test, dic)
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    print "training finished"
    print clf.score(x_test, y_test)
