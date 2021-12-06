from flask import Flask, render_template, request
from collections import Counter
import json
import pandas as pd
data = pd.read_csv('cleaned.csv')

app = Flask(__name__)
s_name = [i.lower() for i in list(data['State'])]

@app.route("/")
def index():
    return render_template("index.html")
@app.route('/state/<string:name>')
def state(name):
    if name.lower() in s_name:
        ids = s_name.index(name.lower())
        result = json.dumps({'State': data.iloc[ids]['State'], 'Age-adjusted incidence rate (cases per 100k)':data.iloc[ids]['Age-Adjusted Incidence Rate([rate note]) - cases per 100,000']})
        return result
    return 'not valid'

@app.route("/info", methods=["GET"])
def analyze():
    usertext = request.args.get("usertext")
    if usertext.lower() in s_name:
        ids = s_name.index(usertext.lower())
        result = json.dumps({'State': data.iloc[ids]['State'], 'Age-adjusted incidence rate (cases per 100k':data.iloc[ids]['Age-Adjusted Incidence Rate([rate note]) - cases per 100,000']})
    else:
        result = 'not valid'
    return render_template("analyze.html", analysis=result, usertext=usertext)


if __name__ == "__main__":
    app.run(debug=True, port = 8007)
