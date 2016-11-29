from flask import Flask

app = Flask(__name__)
app.debug = True


def hi():
    return "from function"


@app.route('/')
def hello_world():
    return hi()
