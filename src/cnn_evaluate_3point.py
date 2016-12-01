import tensorflow as tf
import numpy as np
from cnn_preprocessing import parse_row
# import gensim
# word2vec_model = gensim.models.Word2Vec.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

def word2vec(rows, model, maxLen):
    data = np.zeros((len(rows), maxLen, 300))
    rowIndex = 0
    for row in rows:
        row = parse_row(row)
        wordIndex = 0
        for word in row.split(' '):
            if word in model:
                data[rowIndex, wordIndex, :] = model[word]
            else:
                data[rowIndex, wordIndex, :] = np.random.rand(300) * 0.5 - 0.25
            wordIndex += 1
            if wordIndex == maxLen:
                break
        rowIndex += 1
    return data

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=True)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph('../model/cnn_3point.meta')
        saver.restore(sess, '../model/cnn_3point')

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        def pred(text, model, maxLen):
            x =  word2vec(text, model, maxLen)
            y = sess.run(predictions, {input_x: x, dropout_keep_prob: 1.0})
            return y


#
# if __name__ == '__main__':
#     txt = [u'I thought that @CNN would get better after they failed so badly in their support of Hillary Clinton however, since election, they are worse!', u'BREAKING: Republican State Board Of Elections issue order requiring dismissal of all 52 McCrory election protests. https://t.co/iDkkc0dzXn', u'Trump will soon become the first president who failed to win a majority of the vote either in the general election\u2026 https://t.co/m2UEfPOkjc', u'RT @SHEPMJS: "Dem\'s re-elect Nancy Pelosi leader despite disenchantment over disappointing election results"\nPerfect!Keep Dem, swamp filled\u2026', u"@PcolaBucsfan I'm still in general election mode !! Lol", u"RT @rcooley123: Carl Bernstein Blasts Trump's Unhinged Voter Fraud Claim: 'More Paranoid Than Nixon' | \nhttps://t.co/ZgeVXFL0h1", u'Pelosi holds onto leadership - the same leadership that secured the cluster f*** that was the election. Great idea \U0001f621https://t.co/qvxleTQ8m9', u"RT @FoxNews: .@DrJillStein's Michigan Recount Could Cost Taxpayers $12 Million\nhttps://t.co/y2SCtWIkqO", u'RT @mattyglesias: On November 14, the federal government gave a $32 million tax subsidy to a company owned by Donald &amp; Ivanka Trump. https:\u2026', u'RT @FoxNewsResearch: Cabinet Picks: Entering week 4 since Election Day, #Trump selects his 5th &amp; 6th cabinet nominees, outpacing all #PEOTU\u2026', u'RT @CNN: JUST IN: House Minority Leader Nancy Pelosi defeats Tim Ryan to retain post as top elected Democrat in the House https://t.co/gI6m\u2026', u'https://t.co/hcfbig9WDH', u'Election results kae baad kaafi aasaan hoga## https://t.co/eldm7Ds2kL', u'RT @summerbrennan: Since the election, Russia has moved nuclear weapons closer to our longtime allies in Europe and Asia https://t.co/4AF7C\u2026', u'RT @jmdonsi: Pelosi voter n image of dem voters. Far left coastal libtards. Dems may never win another election!!! https://t.co/VUWmYxFMzO']
#     y = pred(txt, word2vec_model, 24)
#     print y