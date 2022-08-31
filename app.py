from flask import Flask, request, render_template
from utils import sine_cosine_transform, ordinal_encoding, OneHot_encoding,handling_outliers
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
job_mapping = joblib.load('job_mapping.pkl')
scaler = joblib.load('scaler.pkl')

def data_prep(inputs):
    inputs["month_sin"], inputs["month_cos"] = sine_cosine_transform(inputs, feature = "month")
    inputs["job"] = inputs["job"].map(job_mapping)
    inputs["education"] = ordinal_encoding(inputs, feature = "education")
    inputs = OneHot_encoding(inputs, features = ["marital", "default", "housing", "loan", "poutcome", "contact"])
    inputs["pdays"] = inputs.apply(lambda row: handling_outliers(row, "pdays", 365), axis = 1)
    inputs["campaign"] = inputs.apply(lambda row: handling_outliers(row, "campaign", 14), axis = 1)
    inputs["previous"] = inputs.apply(lambda row: handling_outliers(row, "previous", 7), axis = 1)
    inputs["balance"] = inputs.apply(lambda row: handling_outliers(row, "balance", 15000, -1000), axis = 1)
    inputs["age"] = inputs.apply(lambda row: handling_outliers(row, "age", 70), axis = 1)
    data_prepared = scaler.transform(inputs)
    
    return(data_prepared)
    
@app.route("/")
def home():
    return render_template("webpage.html")

@app.route("/predict", methods = ["POST"])
def predict():
    age = int(request.form["Age"])
    job = request.form["Job_type"]
    marital = request.form["Marital_Status"]
    education = request.form["Education"]
    default = request.form["Default"]
    balance = int(request.form["Balance"])
    housing = request.form["Housing"]
    loan = request.form["Loan"]
    day = int(request.form["Day"])
    month = request.form["Month"]
    campaign = int(request.form["Contact_Campaign"])
    previous = int(request.form["Before_Campaign"])
    poutcome = request.form["Prev_outcome"]
    pdays = int(request.form["Pdays"])
    contact = request.form["Contact"]

    inputs = pd.DataFrame({"age": age, "job": job, "marital": marital,"education": education, 
                           "default": default, "balance": balance,"housing": housing, 
                           "loan": loan, "contact": contact, "day": day, "month": month, 
                           "campaign": campaign, "pdays": pdays, "previous": previous,
                           "poutcome": poutcome}, index= [0])
    
    input_prepared = data_prep(inputs)
    prediction = model.predict(input_prepared)
    
    if prediction == 1:
        output = "Success"
    else:
        output = "Failure"
    
    return render_template("webpage.html", prediction_text = "{}".format(output))

if __name__=="__main__":
    app.run(debug=False)
