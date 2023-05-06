from flask import Flask,request,jsonify
import numpy as np
from sklearn import preprocessing
import pandas as pd
import pickle


model = pickle.load(open('sms_classification.sav','rb'))
app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():

  sm=request.form.get('sms')

  list = [sm]
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  li = le.fit_transform(list)

  result = model.predict([li])[0]
  return jsonify({'data': str(result)})

if __name__ == '__main__':
    app.run(debug=True)
