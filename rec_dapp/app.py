from flask import Flask, render_template, jsonify
from recommender_model import *
import pandas as pd

df = pd.read_csv('../rarible_user_data.csv')

def run(df):
    user_vecs, item_vecs = run_model(df)
    make_prediction(customers, products, user_vecs, item_vecs, product_train)
    rec_item = make_prediction(customers, products, user_vecs, item_vecs, product_train)
    return str(rec_item)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    rec_item = run(df)
    return render_template('index.html', prediction_text='The predicted NFT is {}'.format(rec_item))

if __name__ == "__main__":
    app.run()