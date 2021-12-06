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
import random

ns = [round(10**(i/100)) for i in range(400)] 
ts = []
test = [random.randint(1, 50000) for i in range(100)]
for i in tqdm(ns):
    my_tree = Tree()
    for j in range(i):
        my_tree.add(random.randint(1,50000)) 
        
    start = time.time()
    for j in test:
        result = j in my_tree
    ts.append(time.time() - start)
    
    
import plotnine as p9
import pandas as pd

(p9.ggplot(pd.DataFrame({'n': ns, 'time (s)': ts}), p9.aes(x='n', y='time (s)'))
+ p9.geom_point()
 + p9.scale_y_continuous(trans='log10')
+ p9.scale_x_continuous(trans='log10')
)
```
![image](https://user-images.githubusercontent.com/62388643/144774369-42dd5ecb-cfdf-482d-876e-fbba86c830cd.png)






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

# EX3

## Implement a two-dimensional k-nearest neighbors classifier (in particular, do not use sklearn for k-nearest neighbors here): given a list of (x, y; class) data, store this data in a quad-tree (14 points). Given a new (x, y) point and a value of k (the number of nearest neighbors to examine), it should be able to identify the most common class within those k nearest neighbors (14 points).Plot this on a scatterplot, color-coding by type of rice. (3 points)Using 5-fold cross-validation with your k-nearest neighbors implementation, give the confusion matrix for predicting the type of rice with k=1. (4 points) Repeat for k=5. (4 points)


The  k-nearest neighbors classifier and contruction of quad-tree, and also its pratice on rice data are as follows.


```python
import math

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return math.sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
# naive kNN 
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)
```
```python
import numpy as np
import math
import time
import heapq
import pandas as pd
from random import * 
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from statistics import mode 
from tqdm import tqdm
from matplotlib.patches import Rectangle
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.patches as mpatches

class Point:
    def __init__(self,x,y,_cls):
        self.x = x
        self.y = y
        self.cls_ = _cls
class Rect:
    def __init__(self,x,y,width,height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    def contain(self,point):
        return (point.x>=(self.x - self.width)) and (point.x <= (self.x +self.width)) and (point.y>=(self.y-self.height)) and (point.y<=(self.y + self.height))
    def within(self,point,d):
        def dist(x1,y1,x2,y2):
            return math.sqrt((x1-x2)**2+(y1-y2)**2)
        l_x = self.x - self.width
        b_y = self.y - self.height
        h_x = self.x + self.width
        t_y = self.y + self.height
        if point.x>=h_x and point.y>=t_y:
            return dist(h_x,t_y,point.x,point.y)<=d
        elif point.x>=h_x and point.y<=b_y:
            return dist(h_x,b_y,point.x,point.y)<d
        elif point.x>=h_x and point.y<t_y and point.y >b_y:
            return dist(h_x,0,point.x,0)<=d
        elif point.x<=l_x and point.y<=b_y:
            return dist(l_x,b_y,point.x,point.y)<d
        elif point.x<=l_x and point.y>=t_y:
            return dist(l_x,t_y,point.x,point.y)<d
        elif point.x<=l_x and point.y>=b_y:
            return dist(l_x,0,point.x,0)<d
        elif point.x>=l_x and point.x<=h_x and point.y>=t_y:
            return dist(0,t_y,0,point.y)<d
        elif point.x>=l_x and point.x<=h_x and point.y<=b_y:
            return dist(0,b_y,0,point.y)<d
        elif self.contain(point):
            return True
class quadTree:
    def __init__(self,boudary,points,n):
        self.boudary = boudary
        self.capacity = n
        self.isleaf = False
        self.points=points
        self.divided = False
        self.northwest = None
        self.southwest = None
        self.northeast = None
        self.southeast = None
        self.color = {"Cammeo":"ob","Osmancik":"og"}
        self.construct()
    def subdivide(self):
        x = self.boudary.x
        y = self.boudary.y
        width = self.boudary.width
        height = self.boudary.height
        ne = Rect(x + width/2,y+height/2, width/2, height/2)
        nw = Rect(x - width/2,y+height/2, width/2, height/2)
        sw = Rect(x - width/2,y-height/2, width/2, height/2)
        se = Rect(x + width/2,y-height/2, width/2, height/2)
        self.northwest = quadTree(nw,[p for p in self.points if p.x<=x and p.y>=y],self.capacity)
        self.southwest = quadTree(sw,[p for p in self.points if p.x<=x and p.y<y],self.capacity)
        self.northeast = quadTree(ne,[p for p in self.points if p.x>x and p.y>=y],self.capacity)
        self.southeast = quadTree(se,[p for p in self.points if p.x>x and p.y<y],self.capacity)
    def construct(self):
        if len(self.points)<self.capacity:
            self.isleaf = True
            return True
        else:
            if not self.divided:
                self.subdivide()
                self.divided =True
                self.points = []
    def subshow(self,ax):
        ax.add_patch( Rectangle((self.boudary.x - self.boudary.width , self.boudary.y -self.boudary.height), 
                        self.boudary.width*2, self.boudary.height*2, 
                        fc ='none', 
                        ec ='black',  
                        lw = 1))
        for i in self.points:
            ax.plot(i.x,i.y,self.color[i.cls_],markersize = .5)
        if self.divided:
            self.northeast.subshow(ax)
            self.southeast.subshow(ax)
            self.southwest.subshow(ax)
            self.northwest.subshow(ax)                
    def showfig(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)        
        self.subshow(ax)
        plt.xlim([self.boudary.x - self.boudary.width-20, self.boudary.x + self.boudary.width+20]) 
        plt.ylim([self.boudary.y - self.boudary.height-20, self.boudary.y + self.boudary.height+20])
        plt.legend(['Cammeo', 'Osmancik'])
        blue_patch = mpatches.Patch(color='blue', label='Cammeo')
        green_patch = mpatches.Patch(color='green', label='Osmancik')
        plt.legend(handles=[blue_patch, green_patch])
        plt.show()

def knn(quad,pnt,k):     
    res = []
    for p in tqdm(pnt):
        stack = [quad]
        r = (float('-inf'),"")
        pnt_ = []
        while len(stack):
            cur = stack.pop(-1)
            if cur.isleaf and cur.boudary.within(p,-r[0]):
                for i in cur.points:
                        if len(pnt_)<k:
                            heapq.heappush(pnt_,(-math.sqrt((i.x - p.x)**2+(i.y - p.y)**2),i.cls_))
                            r = heapq.nsmallest(1,pnt_)[0]
                        elif math.sqrt((i.x - p.x)**2+(i.y - p.y)**2)<-r[0]:
                            heapq.heappop(pnt_)
                            heapq.heappush(pnt_,(-math.sqrt((i.x - p.x)**2+(i.y - p.y)**2),i.cls_))
                            r = heapq.nsmallest(1,pnt_)[0]
            elif not cur.isleaf:
                if cur.boudary.within(p,-r[0]):
                    if cur.northwest:
                        stack.append(cur.northwest)
                    if cur.southeast:
                        stack.append(cur.southeast)
                    if cur.northeast:
                        stack.append(cur.northeast)
                    if cur.southwest:
                        stack.append(cur.southwest)
        res.append(mode([itr[1] for itr in pnt_]))
    return res 



    
if __name__ == '__main__':
    data = pd.read_excel('Rice_Osmancik_Cammeo_Dataset.xlsx')
    X = data.drop('CLASS', axis=1)
    y = data['CLASS']
    pca = decomposition.PCA(n_components=2)
    my_cols  = data.columns[:(len(data.columns)-1)]
    data_reduced = pca.fit_transform(data[my_cols])
    pc0 = data_reduced[:, 0]
    pc1 = data_reduced[:, 1]
    xlim_min = min(pc0)-.01
    ylim_min = min(pc1)-.01
    xlim_max = max(pc0)+.01
    ylim_max = max(pc1)+.01    
    bound = Rect((xlim_max+xlim_min)/2,(ylim_max+ylim_min)/2,(xlim_max-xlim_min)/2,(ylim_max-ylim_min)/2)
    qt = quadTree(bound,[Point(pc0[k],pc1[k],y.iloc[k]) for k in range(len(pc0))],10)
    qt.showfig()
    for k_near in [1,5]:
        kf = KFold(n_splits=5)
        res_pred = []
        res_true = []
        res_knn = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            data_reduced = pca.fit_transform(X_train[my_cols])
            test_data_reduced = pca.transform(X_test[my_cols])
            pc0 = data_reduced[:, 0]
            test_pc1 = test_data_reduced[:, 1]
            test_pc0 = test_data_reduced[:, 0]
            pc1 = data_reduced[:, 1]       
            xlim_min = min(pc0)-.01
            ylim_min = min(pc1)-.01
            xlim_max = max(pc0)+.01
            ylim_max = max(pc1)+.01

            start_time = time.perf_counter()
            bound = Rect((xlim_max+xlim_min)/2,(ylim_max+ylim_min)/2,(xlim_max-xlim_min)/2,(ylim_max-ylim_min)/2)
            qt = quadTree(bound,[Point(pc0[k],pc1[k],y_train.iloc[k]) for k in range(len(pc0))],n = 10)
            knn_res = knn(qt,[Point(test_pc0[k],test_pc1[k],y_test.iloc[k]) for k in range(len(test_pc0))],k_near)
            end_time = time.perf_counter()
            print(f"quad tree knn time: {end_time - start_time}s")

            start_time = time.perf_counter()
            y_pred = k_nearest_neighbors(list(zip(pc1,pc0,y_train)), list(zip(test_pc1,test_pc0,y_test)), k_near)
            end_time = time.perf_counter()
            print(f"naive knn time: {end_time - start_time}s")


            res_pred = res_pred + knn_res
            res_true = res_true +y_test.to_list()
            res_knn = res_knn+y_pred
        print('#'*40)    
        print('When k = '+str(k_near))
        print('-'*40)
        print('The confusion matrix of quad tree nearest_neighbors')    
        print(confusion_matrix(res_true,res_pred))
        print('-'*40)
        print('The confusion matrix of naive k nearest_neighbors') 
        print(confusion_matrix(res_true,res_knn))
        print('#'*40)
  ```
![image](https://user-images.githubusercontent.com/62388643/144772591-a91c8273-5b32-4b91-8028-ff122b03e8e7.png)
```
########################################
When k = 1
----------------------------------------
The confusion matrix of quad tree nearest_neighbors
[[1224  406]
 [ 414 1766]]
----------------------------------------
The confusion matrix of naive k nearest_neighbors
[[1224  406]
 [ 414 1766]]
########################################
```
```
########################################
When k = 5
----------------------------------------
The confusion matrix of quad tree nearest_neighbors
[[1253  377]
 [ 313 1867]]
----------------------------------------
The confusion matrix of naive k nearest_neighbors
[[1253  377]
 [ 313 1867]]
########################################
```

    

## Comment on what the graph suggests about the effeciveness of using k-nearest neighbors on this 2-dimensional reduction of the data to predict the type of rice. (4 points)
 
The reduced 2-d feature map has obvious decision lines either horizontally or vertically, which could, in other words, be divided into rectangle parts. KNN algorithm, for its inherent characteristic as dividing feature spaces into rectangles, are effective for this data.
 
 
## Provide a brief interpretation of what the confusion matrix results mean.
For k = 1, of all the points, there are 414 entires that are Osmancik but were predicted as Cammeo wrongly and here are 406 entires that are Cammeo
but were predicted as Osmancik wrongly, while 1224 Cammeo were correctly classfied as Cammeo,1766 Osmancik were correctly classfied as Osmancik.
For k = 5, of all the points, there are 313 entires that are Osmancik but were predicted as Cammeo wrongly and here are 377 entires that are Cammeo
but were predicted as Osmancik wrongly, while 1253 Cammeo were correctly classfied as Cammeo,1867 Osmancik were correctly classfied as Osmancik.

Overall when k = 5, the accuracy is higher, this might because in case k = 1, the model is overfitted.
    





