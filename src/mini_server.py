from flask import Flask
from flask import request
from flask import render_template
from flask import url_for
import lstm_chars
import pickle

app = Flask(__name__)
app.debug = True


# initialize all the model
# make sure it is callable
with open('../model/senti_vocab.pkl', 'r') as fp:
    lstm_vocab = pickle.load(fp)
lstm = lstm_chars.Predictor("test5_2point_1.6M_1", lstm_vocab)


@app.route('/')
def home():
    return render_template('index.html')


# put all the model prediction methods here
# 1 stands for positive
# 0 stands for negative
@app.route('/lstm', methods=['POST'])
def predict_for_one_txt():
    ss = request.form['data']
    a = lstm.predict(ss)
    b = str(a[0])
    return b


@app.route('/test', methods=['POST'])
def test_post():
    txt = request.form['data']
    print txt
    return txt

if __name__ == "__main__":
    app.run()
