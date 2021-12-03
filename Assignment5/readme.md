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
```python
from tqdm import tqdm
import numpy as np
import time
ns = [10,100,1000,10000,100000]
ts = []
for n in ns:
    nodes = np.random.random_integers(0,1000, n)
    st_t = time.time()    
    for _ in tqdm(range(100)):
        for node in nodes:
             node in my_tree
    ts.append(time.time()-st_t)
    
    
import plotnine as p9
import pandas as pd

(p9.ggplot(pd.DataFrame({'n': ns, 'time (s)': ts}), p9.aes(x='n', y='time (s)'))
+ p9.geom_point()
 + p9.scale_y_continuous(trans='log10')
+ p9.scale_x_continuous(trans='log10')
)
```
![image](https://user-images.githubusercontent.com/62388643/144682345-71dcd43e-e5df-4044-beb5-247427d64254.png)





## This speed is not free. Provide supporting evidence that the time to setup the tree is O(n log n) by timing it for various sized ns and showing that the runtime lies between a curve that is O(n) and one that is O(n**2).
```python

ns = range(0,400,20)
ts1, ts2, ts3 = [], [], []
import random
import math
# nlogn
for n in tqdm(ns):
    st_t = time.time()


    for _ in range(1000):
        nodes = np.random.random_integers(0, 1000, n)
        tree = Tree()
        for node in nodes:
            tree.add(node)
    ts1.append(time.time() - st_t)
# n
for n in tqdm(ns):
    st_t = time.time()


    for _ in range(1000):


        for i in range(n):
            pass
    ts2.append(time.time() - st_t)
#n^2   
# ns = [int((i/math.log(i))**2) for i in ns]


for n in tqdm(ns):
    st_t = time.time()
    for _ in range(1000):


        for i in range(n):
            for j in range(n):
                pass
    ts3.append(time.time() - st_t)
    
    
import matplotlib.pyplot as plt
plt.plot(ns, ts1)
plt.plot(ns, ts2)
plt.plot(ns, ts3)
plt.legend(['NlogN','N','N^2'])
plt.xlabel('N')
plt.ylabel('Time(s)')
plt.title('Speed comparison')
plt.show()
```
![image](https://user-images.githubusercontent.com/62388643/144682538-852687d2-1631-4b34-a13d-e8ea9cda618d.png)

    
        
    





