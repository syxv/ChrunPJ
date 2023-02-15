from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import pandas as pd


with open("../model/model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    result = {}
    result['pred'] = ""
    result['credit_score'] = 300
    result['gender'] = 0
    result['age'] = 50
    result['tenure'] = 2
    result['balance'] = 120000
    result['products_number'] = 2
    result['credit_card'] = 1
    result['active_member'] = 1
    result['estimated_salary'] = 125000
    result['Germany'] = 0
    result['Spain'] = 0
    result['France'] = 1
    if request.method == "POST":
        result['credit_score'] = request.form["credit_score"]
        result['gender'] = request.form["gender"]
        result['age'] = request.form["age"]
        result['tenure'] = request.form["tenure"]
        result['balance'] = request.form["balance"]
        result['products_number'] = request.form["products_number"]
        result['credit_card'] = request.form["credit_card"]
        result['active_member'] = request.form["active_member"]
        result['estimated_salary'] = request.form["estimated_salary"]
        result['Germany'] = request.form["Germany"]
        result['Spain'] = request.form["Spain"]
        result['France'] = request.form["France"]

        

        x_array = np.array([[
                float(result['credit_score']),
                float(result['gender']),
                float(result['age']), 
                float(result['tenure']),
                float(result['balance']),
                float(result['products_number']),
                float(result['credit_card']),
                float(result['active_member']),
                float(result['estimated_salary']),
                float(result['Germany']), 
                float(result['France']),
                float(result['Spain'])
            ]])
        X = pd.DataFrame(x_array,
                columns = [
                    'credit_score', 'gender', 'age', 'tenure', 'balance', 'products_number',
                    'credit_card', 'active_member', 'estimated_salary', 'Germany', 'France',
                    'Spain'
                ])


        #print(X)
        result['pred'] = model.predict_proba(X)[0][1]
    
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
