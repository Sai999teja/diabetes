from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import joblib

app = Flask(__name__)

from sklearn.ensemble import GradientBoostingClassifier

model = pickle.load(open('diabetes_deploy (1).pkl','rb'))



@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = int(prediction[0])
    

    if (output == 1) :
        return render_template('index.html',pred='There is a high chance of person suffering with diabetes, based on the entered data.')
    else:
        return render_template('index.html',pred='There is a low chance of person suffering with diabetes, based on the entered data.')

if __name__ == '__main__':
    app.run(debug=True)
