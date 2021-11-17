# EX1
## implement a two-dimensional version of the gradient descent algorithm to find optimal choices of a and b. (7 points)Explain how you estimate the gradient given that you cannot directly compute the derivative (3 points), identify any numerical choices -- including but not limited to stopping criteria -- you made (3 points), and justify why you think they were reasonable choices (3 points).
I estimated the partial gradients using (f(a + h, b) - f(a, b))/h and (f(a, b + h) - f(a, b))/h, here h is really small.
I set the learning rate gamma as .1, the h is 1e-4 and the initial a,b as 0.4, 0.2. 
And my stopping criteria is that once the loss does not decrease for a continuous 10 epochs, the gradient descent process stops. The reason I do this is that when the loss does not decrease for a period of time, it indicates that the loss is trapped in a minimum, whether global or local. The resualt shows that the minimum is about 1.000000015 as a,b is  0.7119500000003939, 0.16894999999974553 respectively.

```python
import requests
def err(a, b):
    try:
        return float(requests.get(f'http://ramcdougal.com/cgi-bin/error_function.py?a={a}&b={b}',headers={'User-Agent':'MyScript'}).text)
    except:
        print('bad input')
        return 


h = 1e-4
gamma = 0.1

def GD(a, b):
    loss = err(a, b)
    trial = 0
    while trial < 2:
        a -= gamma*(err(a+h, b)-err(a, b))/h
        b -= gamma*(err(a, b+h)-err(a, b))/h
        if loss > err(a, b):
            loss = err(a, b)
            trial = 0
        else:
            trial+=1
    return loss, a, b
GD(0.4, 0.2)
```
```
>>> (1.000000015, 0.7119500000003939, 0.16894999999974553)
```
            
            

## It so happens that this error function has a local minimum and a global minimum. Find both locations (i.e. a, b values) querying the API as needed (5 points) and identify which corresponds to which (2 point). Briefly discuss how you would have tested for local vs global minima if you had not known how many minima there were. (2 points)
I adjust the initial values by sampling from the normal distribution, and then get two minimuns. One is 1.00000 and the other is 1.10000. 1.00000 is the global minuium and the other is the local minimum. Numerically, the local minimun cannot be smaller than the global one. When testing one whether or not global, I will probably do random initialization ,like Xavier, to get multiple result and compare them. Or, I can adjust the learning rate, like to increase it to help the loss escape from one minimum.

```python
minums = set()
for _ in range(5):
    [a, b] =  np.random.random_sample(2)
    res = GD(a,b)
    minums.add(res)
minums
```
```
>>> {1.10000000113, 1.10000000337, 1.000000015, 1.00000000228, 1.10000000501, 1.10000000077, 1.00000001314, 1.000000005, 1.10000000499, 1.1000000017, 1.10000000483, 1.10000000225, 1.10000000148, 1.10000000073}
```

# EX2
## Modify the k-means code (or write your own) from slides13 to use the Haversine metric and work with our dataset (5 points). Visualize your results with a color-coded scatter plot (5 points); be sure to use an appropriate map projection (i.e. do not simply make x=longitude and y=latitude; 5 points). Use this algorithm to cluster the cities data for k=5, 7, and 15. Run it several times to get a sense of the variation of clusters for each k (share your plots) (5 points); comment briefly on the diversity of results for each k. (5 points)

I run 3 times on each K, and the average of the runtimes are as below result shows. 
Also, I present the figures after clustering by appropriate map projection, which shows that when K get larger, the varience within each cluster gets smaller, and the result becomes more stable. When K is small, the result relys much on the initial centers chosed. However, we could see when K=15, the map gets complicated and hard to interpret. 


```python
import pandas as pd
import plotnine as p9
import random

from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def normalize(series):
    return (series - series.mean()) / series.std()

df['lat_n'] = normalize(df['lat']) 
df['lng_n'] = normalize(df['lng'])
pts = [np.array(pt) for pt in zip(df['lat_n'], df['lng_n'])] 

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
def plot(k,i):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    lats = df['lat'].values
    lngs = df['lng'].values
    clsts = df['cluster'].values
    ax.coastlines()
    plt.title(f'k={k} times n={i+1}')
    ax.scatter(lngs, lats, s=.1,c=clsts, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    plt.show()
    
import time
for k in [5,7,15]:
    ts = []
    for j in range(3):
        t  = time.time()
        centers = random.sample(pts, k)
        old_cluster_ids, cluster_ids = None, [] # arbitrary but different
        while cluster_ids != old_cluster_ids:
            old_cluster_ids = list(cluster_ids)
            cluster_ids = []
            for pt in pts:
                min_cluster = -1
                min_dist = float('inf')
                for i, center in enumerate(centers):
                    dist = haversine(pt[1], pt[0], center[1], center[0])
                    if dist < min_dist:
                        min_cluster = i
                        min_dist = dist
                cluster_ids.append(min_cluster)
        df['cluster'] = cluster_ids
        cluster_pts = [[pt for pt, cluster in zip(pts, cluster_ids) if cluster == match]for match in range(k)]
        centers = [sum(pts)/len(pts) for pts in cluster_pts]
        ts.append(time.time()-t)
        plot(k, j)
        
    print(f'time cost: {np.average(ts)}s')




```
```
>>> time cost: 1.4473318258921306s
>>> time cost: 1.6127390066782634s
>>> time cost: 3.2021297613779702s
```

![1](https://user-images.githubusercontent.com/62388643/141699709-b5532dd2-2213-4e57-a22b-806a8bdb9ce1.png)

![2](https://user-images.githubusercontent.com/62388643/141699730-ac56ef78-1a01-476a-a96d-2972fafc2fd3.png)
![3](https://user-images.githubusercontent.com/62388643/141699731-9a02f9fd-7ebe-4906-8f48-fd445d639bbe.png)

![4](https://user-images.githubusercontent.com/62388643/141699656-15daabf5-df88-41e1-8bff-0056e939daee.png)
![5](https://user-images.githubusercontent.com/62388643/141699657-082ed6d0-4c05-43c5-8af1-c592697a55d8.png)
![6](https://user-images.githubusercontent.com/62388643/141699658-5b7fc1f5-6452-4389-84d7-d026d4c90c36.png)
![7](https://user-images.githubusercontent.com/62388643/141699659-6e192338-15aa-49a9-a8bf-0eba2b68f1de.png)
![8](https://user-images.githubusercontent.com/62388643/141699660-782d5fee-9619-4999-b485-64f4afac4546.png)
![9](https://user-images.githubusercontent.com/62388643/141699661-172554e7-9e6e-45bd-93fa-5ad99634215a.png)

## speedup your code using the multiprocessing module and demonstrate the speedup and discuss how it scales.
I use parallelism when calculating the distance from each point to the center point.
```python
def dis(pt, centers):
    dist = haversine(pt[1], pt[0], centers[1], centers[0])

def multi(k, j):
    #processes = multiprocessing.cpu_count()
    t = time.time()
    centers = random.sample(pts, k)
    old_cluster_ids, cluster_ids = None, []  # arbitrary but different
    while cluster_ids != old_cluster_ids:
        old_cluster_ids = list(cluster_ids)
        cluster_ids = []
        for pt in pts:
            min_cluster = -1
            min_dist = float('inf')
            pool = multiprocessing.Pool(k)

            d = pool.map(partial(dis, pt), centers)
            cluster_ids.append(d.index(min(d)))

    df['cluster'] = cluster_ids

    print(time.time() - t)

```

## EX3
## Implement both (yes, I know, I gave you implementations on the slides, but try to do this exercise from scratch as much as possible) (5 points), time them as functions of n (5 points), and display this in the way you think is best (5 points). Discuss your choices (e.g. why those n and why you're displaying it that way; 5 points) and your results (5 points).

For the naive recursive methods, I choose int from 1-40, and for the lru_cache methods, I choose 1-1000. Because the the former one has O($/alpha^n$) complexity, small ns will not cost massive runtime, and the latter is much faster thanks to the memoization, so it can handle larger ns. I plot loglog figure for the former method's result, because the time is O($\alpha^n$), log on n and time will give a linear plot, which is more clear and interpretable. And I do not use log on y for the latter methods, beacause the computer reports most times as 0. 
```python
def fib1(n):
    
    if n in (1,2):
        return 1
    return fib1(n-1)+fib1(n-2)


from functools import lru_cache
@lru_cache()
def fib3(n):
    # preconditions: n an integer >= 1
    if n in (1, 2):
        return 1
    return fib3(n - 1) + fib3(n - 2)
    
import time
import numpy as np
from tqdm import tqdm
def timeit(function, *args):
    times = []
    for i in range(3):
        start = time.time()
        function(*args)
        times.append(time.time()-start)
    return np.average(times)

ns = range(1, 40)
times1 = [timeit(fib1, n) for n in tqdm(ns)] 
        
        
ns = range(1, 1000)
times2 = [timeit(fib3, n) for n in tqdm(ns)] 



import plotnine as p9
import pandas as pd

(p9.ggplot(pd.DataFrame({'n': ns, 'time (s)': times1}), p9.aes(x='n', y='time (s)'))
+ p9.geom_point()
 + p9.scale_x_continuous()
+ p9.scale_y_continuous(trans='log10')
)

(p9.ggplot(pd.DataFrame({'n': ns, 'time (s)': times2}), p9.aes(x='n', y='time (s)'))
+ p9.geom_point()
 + p9.scale_x_continuous()
+ p9.scale_y_continuous()
)
    
```
![1](https://user-images.githubusercontent.com/62388643/141699075-bfa0665a-20d8-46fa-b121-4a7af9c9f030.png)
![2](https://user-images.githubusercontent.com/62388643/141699076-5c39fa7e-b457-498e-acdc-00ffd658dcea.png)


# EX4
Implement a function that takes two strings and returns an optimal local alignment (6 points) and score (6 points) using the Smith-Waterman algorithm; insert "-" as needed to indicate a gap (this is part of the alignment points). Your function should also take and correctly use three keyword arguments with defaults as follows: match=1, gap_penalty=1, mismatch_penalty=1 (6 points). Here, that is a penalty of one will be applied to match scores for each missing or changed letter.
Test it, and explain how your tests show that your function works. Be sure to test other values of match, gap_penalty, and mismatch_penalty (7 points).

I test the below function with the given samples, and both works well. Then I modified the params to see what happens, and the results are also reasonable by manul check.
```python
import numpy as np

def align(seq1, seq2, match=1, gap_penalty=1, mismatch_penalty=1):
    path = {}
    
    S = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype = 'int')  

    for i in range(0, len(seq1) + 1):
        path['[' + str(i) + ', 0]'] = []
    for i in range(0, len(seq2) + 1):
        path['[0, ' + str(i) + ']'] = []

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                L = S[i - 1, j - 1] + match #match
            else:
                L = S[i - 1, j - 1] - mismatch_penalty #substitute
            
            P = S[i - 1, j] - gap_penalty #gap in seq2
            Q = S[i, j - 1] - gap_penalty #gap in seq1
            S[i, j] = max(L, P, Q, 0)
            path_key = '[' + str(i) + ', ' + str(j) + ']'
            path[path_key] = []
            if L == S[i, j]:
                path[path_key].append('[' + str(i - 1) + ', ' + str(j - 1) + ']')
            if P == S[i, j]:
                path[path_key].append('[' + str(i - 1) + ', ' + str(j) + ']')
            if Q == S[i, j]:
                path[path_key].append('[' + str(i) + ', ' + str(j - 1) + ']')
    end = np.argwhere(S == S.max()) #find the max score

    for i in end: #traceback to find the route
        key = str(list(i))
        value = path[key]
        result = [key]
        traceback(path, S, value, result, seq1, seq2)
    print("score = " + str(S.max()))

def traceback(path, S, value, result, seq1, seq2):
    if value != []:
        i = int((value[0].split(',')[0]).strip('['))
        j = int((value[0].split(',')[1]).strip(']'))        
        key = value[0]
        result.append(key)
        value = path[key]
    if S[i, j] != 0:
        traceback(path, S, value, result, seq1, seq2)
    else:
        x = 0
        y = 0
        s1 = ''
        s2 = ''
        for n in range(len(result)-2, -1, -1):
            point = result[n]
            i = int((point.split(',')[0]).strip('['))
            j = int((point.split(',')[1]).strip(']'))
            if i == x:
                s1 += '-'
                s2 += seq2[j-1]
            elif j == y:
                s1 += seq1[i-1]
                s2 += '-'
            else:
                s1 += seq1[i-1]
                s2 += seq2[j-1]
            x = i
            y = j     
        print("seq1 = {}".format(s1))
        print("seq2 = {}".format(s2))


align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac')
```
```
>>>
seq1 = agacccta-cgt-gac
seq2 = aga-cctagcatcgac
score = 8
```
```python
align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=2)

```
```
>>>
seq1 = gcatcga
seq2 = gcatcga
score = 7
```
```python
align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=2,mismatch_penalty=3)
```

```
>>>
seq1 = gcatcga
seq2 = gcatcga
score = 7
```
```python
align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', match=2, gap_penalty=2,mismatch_penalty=3)
```
```
>>>
seq1 = agacccta-cgt-gac
seq2 = aga-cctagcatcgac
score = 15
```

