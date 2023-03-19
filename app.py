from flask import Flask, request, redirect, render_template, url_for
from flask_sqlalchemy import SQLAlchemy

from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('model/default_model')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    result = model.predict([int(request.form['con']), int(request.form['gap']), int(request.form['pow']), int(request.form['eye']),
                           int(request.form['avk']), int(request.form['conl']), int(request.form['powl']), int(request.form['conr']), int(request.form['powr'])])
    return render_template('index.html', prediction=round(result[0][0]))


if __name__ == "__main__":
    app.run(debug=True)
