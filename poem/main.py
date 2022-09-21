from flask import Flask, render_template, request, send_file
import logging
from gen_text import GenText
from composition import gen_composition, gen_pycode

app = Flask(__name__)
logging.basicConfig(filename='log.txt', level=logging.INFO, encoding="UTF-8",
                    format="%(asctime)s - %(levelname)s - %(message)s")
generator = GenText()


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        seed = request.form['seed']
        gen_type = request.form['radio']
    else:
        seed = ""
        gen_type = ""

    if gen_type == "poem":
        ip_address = request.headers.get("X-Forwarded-For", "")
        logging.info(ip_address)
        try:
            result = generator.gen_poem(seed)
        except BaseException:
            result = []

        if not result:
            logging.info(seed)
            result = ["这个题目有点难，换一个试试吧。"]
        else:
            result = [seed] + [str(i + 1) + ": " + result[i] for i in range(len(result))]
        logging.info(result)
    elif gen_type == "couplet":
        ip_address = request.headers.get("X-Forwarded-For", "")
        logging.info(ip_address)
        try:
            result = generator.gen_couplet(seed)
        except BaseException:
            result = []

        if not result:
            logging.info(seed)
            result = ["这个上联有点难，换一个试试吧。"]
        else:
            result = [str(i + 1) + ": " + result[i] for i in range(len(result))]
        logging.info(result)
    else:
        result = [' ']

    return render_template('shici.html', result=result)

#
# @app.route('/composition/', methods=['POST', 'GET'])
# def composition():
#     if request.method == 'POST':
#         title = request.form['seed']
#     else:
#         title = ""
#
#     if title:
#         result = gen_composition(title)
#         if not result:
#             result = "这个题目有点难，换一个试试吧。"
#         ip_address = request.headers.get("X-Forwarded-For", "")
#         logging.info(ip_address)
#         logging.info(title)
#         logging.info(result)
#     else:
#         result = " "
#
#     return render_template('composition.html', result=result)
#
#
# @app.route('/pycode/', methods=['POST', 'GET'])
# def pycode():
#     if request.method == 'POST':
#         title = request.form['seed']
#     else:
#         title = ""
#
#     if title:
#         result = gen_pycode(title)
#         if not result:
#             result = "这个题目有点难，换一个试试吧。"
#         ip_address = request.headers.get("X-Forwarded-For", "")
#         logging.info(ip_address)
#         logging.info(title)
#         logging.info(result)
#     else:
#         result = " "
#
#     return render_template('pycode.html', title=title, result=result)
#

@app.route('/a_enhanced/')
def image1():
    return send_file('./static/a_enhanced.jpg', mimetype='image/jpg')


@app.route('/a_original/')
def image2():
    return send_file('./static/a_original.jpg', mimetype='image/jpg')


@app.route('/b_enhanced/')
def image3():
    return send_file('./static/b_enhanced.jpg', mimetype='image/jpg')


@app.route('/b_original/')
def image4():
    return send_file('./static/b_original.jpg', mimetype='image/jpg')

@app.route('/026/')
def image5():
    return send_file('./static/026.jpg', mimetype='image/jpg')

if __name__ == '__main__':
    app.run()
