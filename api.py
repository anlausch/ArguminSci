from flask import Flask, request, render_template, redirect, url_for, Response
from logging.handlers import RotatingFileHandler
from time import strftime
import traceback
import logging
from flask_bootstrap import Bootstrap
import serve
from werkzeug.utils import secure_filename
import json

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt'])

class Server:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

server = Server()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Bootstrap(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def connect_spans(result):
    new_result = []
    for sentence in result:
        new_sentence = []
        for i,word_label in enumerate(sentence):
            word_label[1] = assign_style_classes(word_label[1])
            new_sentence.append(word_label)
            if i != len(sentence)-1:
                if word_label[1] == assign_style_classes(sentence[i+1][1]):
                    new_sentence.append([" ", word_label[1]])
                else:
                    new_sentence.append([" ", ""])
        new_result.append(new_sentence)
    return new_result




def assign_style_classes(label):
    # map the inputs to the function blocks
    options = {
        "Token_Label.BEGIN_BACKGROUND_CLAIM": "background_claim",
        "Token_Label.INSIDE_BACKGROUND_CLAIM": "background_claim",
        "Token_Label.BEGIN_OWN_CLAIM": "own_claim",
        "Token_Label.INSIDE_OWN_CLAIM": "own_claim",
        "Token_Label.BEGIN_DATA": "data",
        "Token_Label.INSIDE_DATA": "data",
        "Token_Label.OUTSIDE": "",
        "DRI_Outcome": "outcome",
        "DRI_Approach": "approach",
        "DRI_Challenge": "challenge",
        "DRI_Background": "background",
        "DRI_FutureWork": "future_work",
        "DRI_Unspecified": "",
        "": ""
    }
    return options[label]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not 'api_mode' in request.form:
            text = request.form["text"]
            if text is None or text == "" or text == " ":
                logger.error("No data provided")
                return render_template("index.html", error="Please insert a text before submitting.")
            logger.info("Data: " + json.dumps(text))

            argumentation = server.argumentation.predict(text=text)
            discourse = server.discourse.predict(text=text)

            logger.info("Argumentation: " + json.dumps(argumentation))
            logger.info("Discourse: " + json.dumps(discourse))

            argumentation = connect_spans(argumentation)
            discourse = connect_spans(discourse)

            argumentation = [['<span class="' + word_label[1] + '">' + str(word_label[0]) + '</span>' for j, word_label in enumerate(sentence)] for i, sentence in enumerate(argumentation)]
            discourse = [['<span class="' + word_label[1] + '">' + str(word_label[0]) + '</span>' for j, word_label in
                       enumerate(sentence)] for i, sentence in enumerate(discourse)]

            discourse = [''.join(sentence) for sentence in discourse]
            argumentation = [''.join(sentence) for sentence in argumentation]

            argumentation = ' '.join(argumentation)
            discourse = ' '.join(discourse)
            return render_template("index.html", argumentation=argumentation, discourse=discourse)

        else:
            text = request.form["text"]
            if text is None or text == "" or text == " ":
                logger.error("No data provided")
                return Response(json.dumps({'message': 'Please provide textual data'}), status=400, mimetype='application/json')
            else:
                logger.info("Data: " + text)
                result = serve.predict(text=text, embd_vocab=server.embd_vocab, model=server.model)
                app.logger.info("Result: " + json.dumps(result))
                return Response(json.dumps({'result': result}), status=200, mimetype='application/json')
    except Exception as e:
        return str(e)


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET'])
def to_index():
    return redirect(url_for('index'))


@app.after_request
def after_request(response):
    """ Logging after every request. """
    # This avoids the duplication of registry in the log,
    # since that 500 is already logged via @app.errorhandler.
    if response.status_code != 500:
        ts = strftime('[%Y-%b-%d %H:%M]')
        logger.error('%s %s %s %s %s %s',
                      ts,
                      request.remote_addr,
                      request.method,
                      request.scheme,
                      request.full_path,
                      response.status)
    return response


@app.errorhandler(Exception)
def exceptions(e):
    """ Logging after every Exception. """
    ts = strftime('[%Y-%b-%d %H:%M]')
    tb = traceback.format_exc()
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s',
                  ts,
                  request.remote_addr,
                  request.method,
                  request.scheme,
                  request.full_path,
                  tb)
    return "Internal Server Error", 500


if __name__ == '__main__':
    server = Server(discourse=serve.Model("discourse"), argumentation=serve.Model("argumentation"))

    handler = RotatingFileHandler('./log/app.log', maxBytes=10000, backupCount=3)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    app.run(port=8000)