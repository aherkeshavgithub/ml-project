from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
 
app= Flask(__name__)

with open('logistic_regression.pkl','rb') as model_file:
    model_file_pkl=pickle.load(model_file)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('index.html')

standard_to=StandardScaler()

@app.route('/predict',methods=['GET','POST'])
def predict():

    SepalLengthCm=float(input('SepalLengthCm: '))
    SepalWidthCm=float(input('SepalWidthCm: '))
    PetalLengthCm=float(input('PetalLengthCm: '))
    PetalWidthCm=float(input('PetalWidthCm: '))

    user_data=np.zeros(4)

    user_data[0]=SepalLengthCm
    user_data[1]=SepalWidthCm
    user_data[2]=PetalLengthCm
    user_data[3]=PetalWidthCm

    user_data_input=model_file_pkl.transform(user_data)

    species=model.predict(user_data_input)
