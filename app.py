from flask import Flask, request, render_template

import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("sentiment.pkl", "rb"))
vector = pickle.load(open("vector.pkl", "rb"))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form["text"]

        prediction = model.predict(vector.transform([text]))
        if prediction ==1:

            return render_template('index.html',prediction_text="Your Review is Positive")
        else:
            return render_template('index.html',prediction_text="Your Review is Negative")

    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)








            

        
        
        




    

