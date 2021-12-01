# EX1
The data used is downloaded from https://statecancerprofiles.cancer.gov/incidencerates/index.php, which credits to National Cancer Institute. 
## Perform any necessary data cleaning (e.g. you'll probably want to get rid of the numbers in e.g. "Connecticut(7)" which refer to data source information as well as remove lines that aren't part of the table). Include the cleaned CSV file in your homework submission, and make sure your readme includes a citation of where the original data came from and how you changed the csv file. (5 points)

I first delete the lines of the descriptional heads and notes, which looks like below.

![image](https://user-images.githubusercontent.com/62388643/143939826-3d7031d8-0324-4c21-85a2-016f881a7935.png)
![image](https://user-images.githubusercontent.com/62388643/143940122-5fcc5b32-4d2d-449e-ac6b-414a368cb341.png)

Then I mapped the ```State``` columns to a clean full-text format and stored them to ```cleaned.csv```.
```python
data['State'] = data['State'].map(lambda x : x[:-3])

data.drop("Met Healthy People Objective of ***?",axis =1, inplace = True)

data.to_csv('cleaned.csv')
```

## Using Flask, implement a server that provides three routes
```python
from flask import Flask, render_template, request
import json
import pandas as pd

data = pd.read_csv('cleaned.csv')
s_name = [i.lower() for i in list(data['State'])]
app = Flask(__name__)


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

```
### Homepage:
![image](https://user-images.githubusercontent.com/62388643/143966863-f9a2b707-cd53-4d40-b08e-8eeff61c6738.png)
### API:
```python
import requests
url = 'http://127.0.0.1:8007'
requests.get(url + '/state/new jersey').text
```
```
>>> '{"State": "New Jersey", "Age-Adjusted Incidence Rate([rate note]) - cases per 100,000": 486.7}'
```
### Info page

Capitalization does not matter in this case, so entering ```new jersey``` could also give out the relative information. 
![image](https://user-images.githubusercontent.com/62388643/143957252-0f5fdcc8-f16d-46b6-a555-d42419ef3020.png)
And for those input are not valid, the page will show ```not valid```.
![image](https://user-images.githubusercontent.com/62388643/143967080-b5885e00-6efc-4d08-807f-53de145d1a5a.png)


## Take this exercise one step beyond that which is described above in a way that you think is appropriate, and discuss your extension in your readme.
I use js and css to add a pop-up alert to address the use of this data. And the scripts are in static folder.
```css
.nav{
    background: #bcf;
    height: 65px;
    }
    ul{
        overflow:hidden;
    }
    ul li{
        float:left;
        list-style:none;
        padding: 0 10px;
        line-height: 65px;
    }
    ul li a{
        color:#3a3a;
    }
    img{
        height:1000px;
        width:2000px;
    }
 ```
 ```js
 alert("Data only for personal use")
 ```

![image](https://user-images.githubusercontent.com/62388643/143960699-243550bf-3302-4233-b9ce-ca403e98a6e2.png)


# EX2
## Extend the basic binary tree example from slides2 into a binary search tree that is initially empty (5 points).  Provide an add method that inserts a single numeric value at a time according to the rules for a binary search tree (5 points).Replace the contains method of the tree with the following __contains__ method. The change in name will allow you to use the in operator; e.g. after this change, 55 in my_tree should be True in the above example, whereas 42 in my_tree would be False. Test this. (5 points).
```python
class Tree:
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None
        self.size = 1 if value != None else 0

    def __contains__(self, item):
        if self.value == item:
                return True
        elif self.left and item < self.value:
                return item in self.left
        elif self.right and item > self.value:
            return item in self.right
        else:
            return False
    
    def add(self, item):
        if not self.size:
            self.value = item
        else:
            self._add(item, self)
        self.size += 1

    def _add(self, val, current):
        if val < current.value:
            if current.left:
                self._add(val, current.left)
            else:
                current.left = Tree(val)
        else:
            if current.right:
                self._add(val, current.right)
            else:
                current.right = Tree(val)
    
    def printTree(self):
        # print inorder
        if not self.size:
            return []
        if self.left:
            self.left.printTree() 
        print(self.value)
        if self.right:
            self.right.printTree()
            
            
            
            
my_tree = Tree()
for item in [55, 62, 37, 49, 71, 14, 17]:
     my_tree.add(item)


my_tree.printTree()
 ```
```
>>> 14
17
37
49
55
62
71
```
```python
42 in my_tree
```
```
>>> False
```
```python
71 in my_tree
```
```
>>> True
```
## Using various sizes n of trees (populated with random data) and sufficiently many calls to in (each individual call should be very fast, so you may have to run many repeated tests), demonstrate that in is executing in O(log n) times; on a log-log plot, for sufficiently large n, the graph of time required for checking if a number is in the tree as a function of n should be almost horizontal. (5 points).


## This speed is not free. Provide supporting evidence that the time to setup the tree is O(n log n) by timing it for various sized ns and showing that the runtime lies between a curve that is O(n) and one that is O(n**2).


>Created by statecancerprofiles.cancer.gov on 11/29/2021 3:35 pm.

State Cancer Registries may provide more current or more local data.


Trend
   Rising when 95% confidence interval of average annual percent change is above 0.
   Stable when 95% confidence interval of average annual percent change includes 0.
   Falling when 95% confidence interval of average annual percent change is below 0.

"[rate note] Incidence rates (cases per 100,000 population per year) are age-adjusted to the 2000 US standard population [http://www.seer.cancer.gov/stdpopulations/stdpop.19ages.html] (19 age groups: <1, 1-4, 5-9, ... , 80-84, 85+). Rates are for invasive cancer only (except for bladder cancer which is invasive and in situ) or unless otherwise specified. Rates calculated using SEER*Stat. Population counts for denominators are based on Census populations as modified [https://seer.cancer.gov/popdata/] by NCI. The 1969-2018 US Population Data File [https://seer.cancer.gov/popdata/] is used for SEER and NPCR incidence rates."
"[trend note] Incidence data come from different sources. Due to different years of data availability, most of the trends are AAPCs based on APCs but some are APCs calculated in SEER*Stat. Please refer to the source for each area for additional information."

Rates and trends are computed using different standards for malignancy. For more information see malignant.html.

"^ All Stages refers to any stage in the Surveillance, Epidemiology, and End Results (SEER) summary stage [ https://seer.cancer.gov/tools/ssm/ ]."
"[rank note]Results presented with the CI*Rank statistics help show the usefulness of ranks. For example, ranks for relatively rare diseases or less populated areas may be essentially meaningless because of their large variability, but ranks for more common diseases in densely populated regions can be very useful. More information about methodology can be found on the CI*Rank website."
*** No Healthy People 2020 Objective for this cancer.
Healthy People 2020 Objectives [ https://www.healthypeople.gov/ ]provided by the Centers for Disease Control and Prevention [ https://www.cdc.gov ]. 

"* Data has been suppressed to ensure confidentiality and stability of rate estimates.  Counts are suppressed if fewer than 16 records were reported in a specific area-sex-race category. If an average count of 3 is shown, the total number of cases for the time period is 16 or more which exceeds suppression threshold (but is rounded to 3)."

"1 Source: National Program of Cancer Registries [ https://www.cdc.gov/cancer/npcr/index.htm ] and Surveillance, Epidemiology, and End Results [ http://seer.cancer.gov ] SEER*Stat Database (2001-2018) - United States Department of Health and Human Services, Centers for Disease Control and Prevention and National Cancer Institute. Based on the 2020 submission."
"5 Source: National Program of Cancer Registries [ https://www.cdc.gov/cancer/npcr/index.htm ] and Surveillance, Epidemiology, and End Results [ http://seer.cancer.gov ] SEER*Stat Database (2001-2018) - United States Department of Health and Human Services, Centers for Disease Control and Prevention and National Cancer Institute. Based on the 2020 submission."
"6 Source: National Program of Cancer Registries SEER*Stat Database (2001-2018) - United States Department of Health and Human Services, Centers for Disease Control and Prevention (based on the 2020 submission).  [ https://www.cdc.gov/cancer/npcr/index.htm ]"
7 Source: SEER November 2020 submission.
"8 Source: Incidence data provided by the SEER Program. ( http://seer.cancer.gov ) AAPCs are calculated by the Joinpoint Regression Program ( https://surveillance.cancer.gov/joinpoint/ ) and are based on APCs. Data are age-adjusted to the 2000 US standard population ( http://www.seer.cancer.gov/stdpopulations/single_age.html ) (19 age groups: <1, 1-4, 5-9, ... , 80-84,85+). Rates are for invasive cancer only (except for bladder cancer which is invasive and in situ) or unless otherwise specified. Population counts for denominators are based on Census populations as modifed by NCI. The 1969-2018 US Population Data ( http://seer.cancer.gov/popdata/ ) File is used with SEER November 2020 data. "


"Interpret Rankings provides insight into interpreting cancer incidence statistics.  When the population size for a denominator is small, the rates may be unstable.  A rate is unstable when a small change in the numerator (e.g., only one or two additional cases) has a dramatic effect on the calculated rate."

Data for United States does not include Puerto Rico.


