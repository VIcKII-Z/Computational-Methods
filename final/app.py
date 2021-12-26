# import libraries
import pandas as pd
import seaborn as sns
import numpy as np
from statistics import mode
from matplotlib import pyplot as plt
from flask import Flask, render_template, redirect, request, url_for, jsonify, send_file
import requests
import jinja2
import tempfile
from pandas.plotting import table
import uuid

# import data

data = pd.read_csv("ventilator-pressure-prediction/train.csv")

analysis_options = ['R', 'C', 'u_in', 'u_out', 'pressure']

# implement a server that provides three routes using flask
app = Flask(__name__)

# the index/homepage written in HTML which prompts the user to select an item for analysis
# and provides a button, which passes this information to /info
@app.route("/")
def index():
    return render_template("index.html", analysis_options=analysis_options, gif='/static/duck.gif', img='/static/site.png')


# select type of drug used and show stats
@app.route("/mono/<varible>")
def show_drug_type_stats(varible):
    y = data[data['breath_id'] == 3][varible]
    x = data[data['breath_id']==3]['time_step']
    
    plt.figure()
    plt.scatter(x, y)
    plt.ylabel(varible)
 
    path = 'output/mono.png'
    plt.savefig(path)
    return send_file(path, mimetype='image/png')


@app.route("/info", methods=["GET"])
def info():
    analysis_type = request.args.get("Actions")

    # check if selected option is valid
    if str(analysis_type) in analysis_options:
        # initialize variables
        
        fp = "/"
        tc = ""
        mc = ""
        fc = ""
    
        var  = str(analysis_type)
            
        
        fp = f'/mono/{var}'
        # upating strings
        tc = "Sample ID :3 "
        mc = f"Max of {analysis_type} is {max(data[data['breath_id'] == 3][var])} " 
        fc = f"Min of {analysis_type} is {min(data[data['breath_id'] == 3][var])} " 
           
       
        return render_template("info.html", analysis_type=analysis_type, fp=fp, tc=tc, mc=mc, fc=fc)

    # analysis type not included in choices: failure
    return render_template("failure.html")

if __name__ == "__main__":
    app.run(debug=True)