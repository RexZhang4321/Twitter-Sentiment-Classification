import twitter

if __name__ == "__main__":
    api = twitter.Api(consumer_key="",
                      consumer_secret="",
                      access_token_key="-",
                      access_token_secret="")
    res = api.GetSearch(term="election", count=15)
    print res[0].AsDict()['text']
