from flask import Flask
from flask import request
from flask import render_template
from flask import url_for
import lstm_chars
import pickle
import twitter_service
import json

app = Flask(__name__)
app.debug = True


# initialize all the model
# make sure it is callable
with open('../model/senti_vocab.pkl', 'r') as fp:
    lstm_vocab = pickle.load(fp)
lstm2 = lstm_chars.Predictor("test5_2point_1.6M_1", lstm_vocab)
# lstm3 = lstm_chars.Predictor("test5_3point_15", lstm_vocab)

tws = twitter_service.TwitterServ()


@app.route('/')
def home():
    return render_template('index.html')


# put all the model prediction methods here
# 1 stands for positive
# 0 stands for negative
@app.route('/lstm', methods=['POST'])
def predict_for_one_txt():
    ss = request.form['data']
    a = lstm2.predict(ss)
    b = str(a[0])
    return b


@app.route('/predlist', methods=['POST'])
def test_post():
    qry = request.form['query']
    print qry
    txt = tws.get_list(qry)
    # txt = [u'I thought that @CNN would get better after they failed so badly in their support of Hillary Clinton however, since election, they are worse!', u'BREAKING: Republican State Board Of Elections issue order requiring dismissal of all 52 McCrory election protests. https://t.co/iDkkc0dzXn', u'Trump will soon become the first president who failed to win a majority of the vote either in the general election\u2026 https://t.co/m2UEfPOkjc', u'RT @SHEPMJS: "Dem\'s re-elect Nancy Pelosi leader despite disenchantment over disappointing election results"\nPerfect!Keep Dem, swamp filled\u2026', u"@PcolaBucsfan I'm still in general election mode !! Lol", u"RT @rcooley123: Carl Bernstein Blasts Trump's Unhinged Voter Fraud Claim: 'More Paranoid Than Nixon' | \nhttps://t.co/ZgeVXFL0h1", u'Pelosi holds onto leadership - the same leadership that secured the cluster f*** that was the election. Great idea \U0001f621https://t.co/qvxleTQ8m9', u"RT @FoxNews: .@DrJillStein's Michigan Recount Could Cost Taxpayers $12 Million\nhttps://t.co/y2SCtWIkqO", u'RT @mattyglesias: On November 14, the federal government gave a $32 million tax subsidy to a company owned by Donald &amp; Ivanka Trump. https:\u2026', u'RT @FoxNewsResearch: Cabinet Picks: Entering week 4 since Election Day, #Trump selects his 5th &amp; 6th cabinet nominees, outpacing all #PEOTU\u2026', u'RT @CNN: JUST IN: House Minority Leader Nancy Pelosi defeats Tim Ryan to retain post as top elected Democrat in the House https://t.co/gI6m\u2026', u'https://t.co/hcfbig9WDH', u'Election results kae baad kaafi aasaan hoga## https://t.co/eldm7Ds2kL', u'RT @summerbrennan: Since the election, Russia has moved nuclear weapons closer to our longtime allies in Europe and Asia https://t.co/4AF7C\u2026', u'RT @jmdonsi: Pelosi voter n image of dem voters. Far left coastal libtards. Dems may never win another election!!! https://t.co/VUWmYxFMzO']
    print txt
    res = lstm2.predict(txt)
    res_lst = []
    for i in range(0, len(txt)):
        res_lst.append({
            "text": txt[i],
            "pred": res[i]
        })
    return json.dumps(res_lst)

if __name__ == "__main__":
    app.run()
