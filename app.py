from flask import Flask, render_template, request
# import jsonify
import requests
import pickle
import json
import numpy as np
import sklearn
# from sklearn.preprocessing import StandardScaler

# with open("artifacts/logistic_regression.pkl","rb") as model_file:
#     model_file_pkl=pickle.load(model_file)
    

model_file_pkl=pickle.load(open("artifacts/logi_reg.pkl","rb"))

with open("artifacts/columns_names.json","r") as json_file:
    col_name=json.load(json_file)
# print(col_name)
col_name_list=col_name['col_name']
# print(col_name_list)

app= Flask(__name__)

# @app.route('/')
# def home():
#     return "Default API"

@app.route('/')
def index():
    return render_template('index.html')

# standard_to=StandardScaler()

@app.route('/predict',methods=['GET','POST'])
def predict():
    data=request.form
    user_data=np.zeros(len(col_name_list))
    # user_data=np.zeros(4)


    # SepalLengthCm=data['SepalLengthCm']
    # SepalWidthCm=data['SepalWidthCm']
    # PetalLengthCm=data['PetalLengthCm']
    # PetalWidthCm=data['PetalWidthCm']
    
    
    user_data[0]=float(data['SepalLengthCm'])
    user_data[1]=float(data['SepalWidthCm'])
    user_data[2]=float(data['PetalLengthCm'])
    user_data[3]=float(data['PetalWidthCm'])

    print(user_data)
    result=model_file_pkl.predict([user_data])
    
    print(result[0])

    if result[0]==0:
        final_result='Iris-setosa'
    
    elif result[0]==1:
        final_result='Iris-versicolor'
    
    elif result[0]==2:
        final_result='Iris-virginica'
    else:
        final_result='Entered values are not in range'

    return render_template ("index.html", prediction=final_result) 

if __name__=="__main__":
    app.run(host="0.0.0.0",port="8080",debug=True)   ### for  deployment host='0.0.0.0'