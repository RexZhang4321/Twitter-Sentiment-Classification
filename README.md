# Twitter sentiment classification
UVa 16fall CS6501-004 Natural Language Processing

## Requirement

###python
``python2.7`` required.

###model
``theano``,
``leasage``,
``pandas``,
``numpy``,
``scikit learn``

###web
``flask``
>> ``jquery`` and ``bootstrap`` included as link

## How to run

Make sure you have installed all the packages.

### Add theano running flags

To run, you need first to configure theano flags in ``~/.theanorc``.
>> Though there are a lot of other ways to config, this is the easiest and painless way.

In ``~/.theanorc``, add following
```
[gcc]
cxxflags= -march=core2

[global]
device = cpu
floatX = float32
```

## Add twitter api keys

The demo will pull twitter tweets, so you need to configure your own twitter account api to make it work.
>> See https://apps.twitter.com/ for more details to get keys.

Create a python file named ``twitter_config.py``, write following codes:
```
twt_config = {
    'consumer_key': ---YOURS---,
    'consumer_secret': ---YOURS---,
    'access_token_key': ---YOURS---,
    'access_token_secret': ---YOURS---
}
```
Replace the '---YOURS---' to your corresponding keys, remember to quote them.

## Configure flask

Last step, before running, enter ``./src`` directory in your terminal, enter ``export FLASK_APP=mini_server.py``, then enter ``flask run`` to run the server.

Open a browser, enter ``127.0.0.1:5000`` to enjoy!
>> You can use other way to deploy, however this is quite easy and simple just for a try.