from flask import Flask
from flask import request
from flask import render_template
from flask import url_for
import gensim
import lstm_chars
import pickle
import twitter_service
import json
import cnn_evaluate_3point
import cnn_evaluate_2point
app = Flask(__name__)
app.debug = True


# initialize all the model
# make sure it is callable
word2vec_model = gensim.models.Word2Vec.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

with open('../model/senti_vocab.pkl', 'r') as fp:
    lstm_vocab = pickle.load(fp)
lstm2 = lstm_chars.Predictor("test5_2point_1.6M_1", lstm_vocab, 2)
lstm3 = lstm_chars.Predictor("test5_3point_15", lstm_vocab, 3)

tws = twitter_service.TwitterServ()


@app.route('/')
def home():
    return render_template('index.html')


# put all the model prediction methods here
# 1 stands for positive
# 0 stands for negative
@app.route('/predone', methods=['POST'])
def predict_for_one_txt():
    # tweet to be classified
    ss = request.form['tweet']
    ss = json.loads(ss)
    model_type = request.form['model_type']
    pred = []
    ''' the result should be organized like this format
    [
        {
            'model': 'LSTM',
            'pred': 0
        },
        ...
    ]
    '''
    if int(model_type) == 2:
        # put 2 points prediction here
        pred.append({
            'pred': int(lstm2.predict(ss)[0]),
            'model': 'lstm'
        })
    else:
        # put 3 points prediction here
        pred.append({
            'pred': int(lstm3.predict(ss)[0]),
            'model': 'lstm'
        })
    return json.dumps(pred)

@app.route('/predlist_cnn', methods=['POST'])
def predict_for_list_cnn():
    # get n items to search
    count = request.form['count']
    # get query to search
    qry = request.form['query']
    # get twitters using this query
    txt = tws.get_list(qry, cnt=15)
    # a test data
    # txt = [u'I thought that @CNN would get better after they failed so badly in their support of Hillary Clinton however, since election, they are worse!', u'BREAKING: Republican State Board Of Elections issue order requiring dismissal of all 52 McCrory election protests. https://t.co/iDkkc0dzXn', u'Trump will soon become the first president who failed to win a majority of the vote either in the general election\u2026 https://t.co/m2UEfPOkjc', u'RT @SHEPMJS: "Dem\'s re-elect Nancy Pelosi leader despite disenchantment over disappointing election results"\nPerfect!Keep Dem, swamp filled\u2026', u"@PcolaBucsfan I'm still in general election mode !! Lol", u"RT @rcooley123: Carl Bernstein Blasts Trump's Unhinged Voter Fraud Claim: 'More Paranoid Than Nixon' | \nhttps://t.co/ZgeVXFL0h1", u'Pelosi holds onto leadership - the same leadership that secured the cluster f*** that was the election. Great idea \U0001f621https://t.co/qvxleTQ8m9', u"RT @FoxNews: .@DrJillStein's Michigan Recount Could Cost Taxpayers $12 Million\nhttps://t.co/y2SCtWIkqO", u'RT @mattyglesias: On November 14, the federal government gave a $32 million tax subsidy to a company owned by Donald &amp; Ivanka Trump. https:\u2026', u'RT @FoxNewsResearch: Cabinet Picks: Entering week 4 since Election Day, #Trump selects his 5th &amp; 6th cabinet nominees, outpacing all #PEOTU\u2026', u'RT @CNN: JUST IN: House Minority Leader Nancy Pelosi defeats Tim Ryan to retain post as top elected Democrat in the House https://t.co/gI6m\u2026', u'https://t.co/hcfbig9WDH', u'Election results kae baad kaafi aasaan hoga## https://t.co/eldm7Ds2kL', u'RT @summerbrennan: Since the election, Russia has moved nuclear weapons closer to our longtime allies in Europe and Asia https://t.co/4AF7C\u2026', u'RT @jmdonsi: Pelosi voter n image of dem voters. Far left coastal libtards. Dems may never win another election!!! https://t.co/VUWmYxFMzO']
    # 2 points or 3 points
    model_type = request.form['model_type']
    res_lst = []
    ''' the result should be organized like this format
    [
        {
            'text': tweet1,
            'predict': [
                {
                    'model': LSTM,
                    'pred': 0
                },
                {
                    'model': CNN,
                    'pred': 1
                },
                ...
            ]
        },
        ...
    ]
    '''
    if int(model_type) == 2:
        # put your 2 points prediction here
        cnn_pred = cnn_evaluate_2point.pred(txt, word2vec_model, 30)
        for i in range(0, len(txt)):
            pred_lst = []
            pred_lst.append({
                "model": 'lstm',
                "pred": cnn_pred[i]
            })
            res_lst.append({
                'text': txt[i],
                'predict': pred_lst
            })
    else:
        # put your 3 points prediction here
        cnn_pred = cnn_evaluate_3point.pred(txt, word2vec_model, 24)
        for i in range(0, len(txt)):
            pred_lst = []
            pred_lst.append({
                "model": 'lstm',
                "pred": cnn_pred[i]
            })
            res_lst.append({
                'text': txt[i],
                'predict': pred_lst
            })
    return json.dumps(res_lst)

@app.route('/predlist', methods=['POST'])
def predict_for_list():
    # get n items to search
    count = request.form['count']
    # get query to search
    qry = request.form['query']
    # get twitters using this query
    txt = tws.get_list(qry, cnt=15)
    # a test data
    # txt = [u'I thought that @CNN would get better after they failed so badly in their support of Hillary Clinton however, since election, they are worse!', u'BREAKING: Republican State Board Of Elections issue order requiring dismissal of all 52 McCrory election protests. https://t.co/iDkkc0dzXn', u'Trump will soon become the first president who failed to win a majority of the vote either in the general election\u2026 https://t.co/m2UEfPOkjc', u'RT @SHEPMJS: "Dem\'s re-elect Nancy Pelosi leader despite disenchantment over disappointing election results"\nPerfect!Keep Dem, swamp filled\u2026', u"@PcolaBucsfan I'm still in general election mode !! Lol", u"RT @rcooley123: Carl Bernstein Blasts Trump's Unhinged Voter Fraud Claim: 'More Paranoid Than Nixon' | \nhttps://t.co/ZgeVXFL0h1", u'Pelosi holds onto leadership - the same leadership that secured the cluster f*** that was the election. Great idea \U0001f621https://t.co/qvxleTQ8m9', u"RT @FoxNews: .@DrJillStein's Michigan Recount Could Cost Taxpayers $12 Million\nhttps://t.co/y2SCtWIkqO", u'RT @mattyglesias: On November 14, the federal government gave a $32 million tax subsidy to a company owned by Donald &amp; Ivanka Trump. https:\u2026', u'RT @FoxNewsResearch: Cabinet Picks: Entering week 4 since Election Day, #Trump selects his 5th &amp; 6th cabinet nominees, outpacing all #PEOTU\u2026', u'RT @CNN: JUST IN: House Minority Leader Nancy Pelosi defeats Tim Ryan to retain post as top elected Democrat in the House https://t.co/gI6m\u2026', u'https://t.co/hcfbig9WDH', u'Election results kae baad kaafi aasaan hoga## https://t.co/eldm7Ds2kL', u'RT @summerbrennan: Since the election, Russia has moved nuclear weapons closer to our longtime allies in Europe and Asia https://t.co/4AF7C\u2026', u'RT @jmdonsi: Pelosi voter n image of dem voters. Far left coastal libtards. Dems may never win another election!!! https://t.co/VUWmYxFMzO']
    # 2 points or 3 points
    model_type = request.form['model_type']
    res_lst = []
    ''' the result should be organized like this format
    [
        {
            'text': tweet1,
            'predict': [
                {
                    'model': LSTM,
                    'pred': 0
                },
                {
                    'model': CNN,
                    'pred': 1
                },
                ...
            ]
        },
        ...
    ]
    '''
    if int(model_type) == 2:
        # put your 2 points prediction here
        lstm_pred = lstm2.predict(txt)
        for i in range(0, len(txt)):
            pred_lst = []
            pred_lst.append({
                "model": 'lstm',
                "pred": lstm_pred[i]
            })
            res_lst.append({
                'text': txt[i],
                'predict': pred_lst
            })
    else:
        # put your 3 points prediction here
        lstm_pred = lstm3.predict(txt)
        for i in range(0, len(txt)):
            pred_lst = []
            pred_lst.append({
                "model": 'lstm',
                "pred": lstm_pred[i]
            })
            res_lst.append({
                'text': txt[i],
                'predict': pred_lst
            })
    return json.dumps(res_lst)


@app.route('/test', methods=['GET', 'POST'])
def test():
    print "success"
    return 'success'

if __name__ == "__main__":
    app.run()
