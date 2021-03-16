from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('random_forest_classifier_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity=float(request.form['volatile_acidity'])
        citric_acid=float(request.form['citric_acid'])
        residual_sugar=float(request.form['residual_sugar'])
        chlorides=float(request.form['chlorides'])
        free_sulfur_dioxide=float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide=float(request.form['total_sulfur_dioxide'])	
        density=float(request.form['density'])
        pH=float(request.form['pH'])
        sulphates=float(request.form['sulphates'])
        alcohol=float(request.form['alcohol'])
        predictions=model.predict([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]])
        output=round(predictions[0],1)
        if output>=3:
            return render_template('index.html',prediction_texts="wine quality is .{}format(output)")
        else:
            return render_template('index.html',prediction_text="your wine quality is bad")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

