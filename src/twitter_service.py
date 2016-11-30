import twitter
import twitter_config


class TwitterServ():

    def __init__(self):
        self.api = twitter.Api(
            consumer_key=twitter_config.twt_config['consumer_key'],
            consumer_secret=twitter_config.twt_config['consumer_secret'],
            access_token_key=twitter_config.twt_config['access_token_key'],
            access_token_secret=twitter_config.twt_config['access_token_secret'])

    def get_list(self, query, cnt=15):
        res = self.api.GetSearch(term=query, count=cnt)
        data = []
        for sts in res:
            data.append(sts.AsDict()['text'])
        return data


if __name__ == "__main__":
    ts = TwitterServ()
    a = ts.get_list("election")
    print a
    print len(a)
