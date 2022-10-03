from flask import Flask, request, jsonify
from flask_cors import cross_origin
import logging

from gen_text import GenText

app = Flask(__name__)
logging.basicConfig(filename='log.txt', level=logging.INFO, encoding="UTF-8",
                    format="%(asctime)s - %(levelname)s - %(message)s")
generator = GenText()

@app.route('/poem/', methods=['POST', 'GET'])
@cross_origin(["https://qizhen.xyz", "https://www.qizhen.xyz"])
def gen_poem():
    seed = request.form['seed']
    gen_type = request.form['radio']

    ip_address = request.headers.get("X-Forwarded-For", "")
    logging.info(request.remote_addr)

    if gen_type == "poem":
        try:
            result = generator.gen_poem(seed)
        except BaseException:
            result = []

        if not result:
            logging.info(seed)
            result = ["这个题目有点难，换一个试试吧。"]

        logging.info([seed] + [str(i + 1) + ": " + result[i] for i in range(len(result))])
    elif gen_type == "couplet":
        try:
            result = generator.gen_couplet(seed)
        except BaseException:
            result = []

        if not result:
            logging.info(seed)
            result = ["这个上联有点难，换一个试试吧。"]

        logging.info([str(i + 1) + ": " + result[i] for i in range(len(result))])
    else:
        result = [' ']

    response = jsonify(result)
    return response



if __name__ == '__main__':
    app.run()
