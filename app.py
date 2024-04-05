#!/usr/bin/env python
# coding: utf-8

# Flask app for heart predictive model

# In[ ]:


import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

#Prediction function
def HeartDiseasePredictor(predict_list):
    to_predict  = np.array(predict_list).reshape(1,13)
    loaded_model = pickle.load(open("heart_disease_model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        predict_list = list(predict_list.values())
        try:
            predict_list = [int(float(x)) for x in predict_list]  
        except ValueError:
            return "Invalid input. Please provide valid integer or float values."

        result = HeartDiseasePredictor(predict_list)
        if int(result) == 1:
            prediction = 'This person is suffering from heart disease..'
        else:
            prediction = 'This person is not suffering from heart disease..'
        return render_template("index.html", prediction=prediction)

 
if __name__=="__main__":
    app.run(port=5000)


# In[ ]:

