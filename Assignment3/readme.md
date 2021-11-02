# Assign 3

# Ex1
## Use the requests module (or urllib) to use the Entrez API (see slides8) to identify the PubMed IDs for 1000 Alzheimers papers from 2019 and for 1000 cancer papers from 2019. (9 points)
```python
import requests
import xml

def search(item):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={item}&retmode=xml&retmax=1000"
    r = requests.get(url)
    doc = xml.dom.minidom.parseString(r.text)
    id_list = []
    ids = doc.getElementsByTagName("Id")
    for i in range(len(ids)):
        id_list.append(ids[i].childNodes[0].data) 
    return id_list   


ids1 = search("Alzheimers+AND+2019[pdat]")
ids2 = search("Cancer+AND+2019[pdat]")
```
## Use the Entrez API via requests/urllib to pull the metadata for each such paper found above (both cancer and Alzheimers) (and save a JSON file storing each paper's title, abstract, MeSH terms (DescriptorName inside of MeshHeading), and the query that found it that is of the general form: (12 points)
I add access monitor in case some of the urls cannot be fetched.
### Be sure to store all parts. You could do this in many ways, from using a dictionary or a list or simply concatenating with a space in between. 
I simply use string concatenation. I first tried to use dict to strore the abstract corresponding to their labels, such as {'BACKGROUND':xxx,...}. 
However, I found the labels do not count much to our analysis in this case, so I choose to concatenate the string. The strength is that it is super convienient for text analysis thanks to its simple formatting as strings. It also has weakness like the memory cost and infexibility.
```python
import requests
import xml.dom.minidom as m
import time
def get_info(pmid):
    # access monitor
    try:
        r = requests.get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id={pmid}" )
        time.sleep(1)
    except:
        print(f'get {pmid} failed')
        
    doc = m.parseString(r.text)
    abstract, mesh = '', []
    titles = doc.getElementsByTagName("Title") 
    if not titles:
        titles = doc.getElementsByTagName('BookTitle')
    abstracts = doc.getElementsByTagName("AbstractText") 
    meshes = doc.getElementsByTagName("DescriptorName") 

    if  not abstracts:
        abstract = ''
    else:
        for abst in abstracts:     
            tree = et.fromstring(abst.toxml())
            abstract+= et.tostring(tree, method='text').decode()
            abstract+='\n'
    if not meshes:
        mesh = meshes
    else:
        for mes in meshes:
            for node in mes.childNodes:               
                mesh.append(node.data) 


                    
    title = titles[0].childNodes[0].wholeText


   
    return title, abstract, mesh
```
```python
import json
j ={}
for pmid in ids1+ids2:
    j[pmid] = {}
    title, abstract, mesh = get_info(pmid)
    if pmid in ids1 and pmid in ids2:
        
        
        query = 'Alzheimer&Cancer'
    elif pmid in ids1:
        query = 'Alzheimer'
    else:
        query = 'Cancer'
    j[pmid]['ArticleTitle'], j[pmid]['AbstractText'], j[pmid]['query'], j[pmid]['mesh'] = title, abstract, query, mesh

with open('articles2019_n.json', 'w', encoding='utf8') as f:
    json.dump(j, f)
```

## There are of course many more papers of each category, but is there any overlap in the two sets of papers that you identified?
```python
set(ids1)&set(ids2)
```
```
>>> {'32322464', '32501203'}
```
# EX2
## What fraction of the Alzheimer's papers have no MeSH terms? (2 points) What fraction of the cancer papers have no MeSH terms? (2 points) Comment on how the fractions compare. (1 point; i.e. if they're essentially the same, do you think that's a coincidence? If they're different, do you have any theories why?)
The fraction of Cancer(0.695) is higher than that of Alzheimers(0.152).
Since MeSH (Medical Subject Headings) is the National Library of Medicine's controlled vocabulary thesaurus, used for indexing articles for the MEDLINE®/PubMED® database and each article citation is associated with a set of MeSH terms that describe the content of the citation, I guess the cancer convers a much larger field  and has much more flexible and broad content so that the database cannot precisely index it or cannot find many existing index to MeSH it.

```python
count1, count2 = 0, 0
for a in ar:
    
    if not ar[a]['mesh']:
        if ar[a]['query'] == 'Alzheimer' or ar[a]['query'] == 'Alzheimer&Cancer':
            
            count1+=1
        elif ar[a]['query'] == 'Cancer' or ar[a]['query'] == 'Alzheimer&Cancer':
            
            count2+=1
count1/1000, count2/1000
```
```
>>> (0.152, 0.695)
```

## What are the 10 most common MeSH terms for the Alzheimer's papers whose metadata you found in Exercise 1? (2 points) Provide a graphic illustrating their relative frequency. (3 points)
```python
mesh_dict = {}
for art in ar:
    if ar[art]['query'] =='Alzheimer' or ar[art]['query'] =='Alzheimer&Cancer':
        meshes = al[art]['mesh']
        if meshes:
            for mesh in meshes:
                mesh_dict[mesh] = mesh_dict.get(mesh, 0) + 1

top10 = sorted(mesh_dict.items(), key = lambda x: x[1], reverse=True)[:10]
top10
```
```
>>> 
[('Alzheimer Disease', 704),
 ('Humans', 701),
 ('Male', 361),
 ('Female', 308),
 ('Aged', 292),
 ('Animals', 285),
 ('Amyloid beta-Peptides', 213),
 ('Brain', 193),
 ('Aged, 80 and over', 173),
 ('Cognitive Dysfunction', 163)]
```
```python
import matplotlib.pyplot as plt

mesh10, frequency = zip(*top10)
plt.barh(mesh10, frequency)
plt.title('top10 MeSH frequency')
plt.xlabel('frequency')
plt.show()
```
![1](https://user-images.githubusercontent.com/62388643/139727155-8585b2ad-787d-4082-85d5-0a1f28c2c7f5.png)


## What are the 10 most common MeSH terms for the cancer papers whose metadata you found in Exercise 1? (2 points) Provide a graphic illustrating their relative frequency.
```python
mesh_dict1 = {}
for art in ar:
    if ar[art]['query'] =='Cancer' or ar[art]['query'] =='Alzheimer&Cancer':
        meshes = ar[art]['mesh']
        if meshes:
            for mesh in meshes:
                mesh_dict1[mesh] = mesh_dict1.get(mesh, 0) + 1

top10_ = sorted(mesh_dict1.items(), key = lambda x: x[1], reverse=True)[:10]
top10_
```
```
>>>
[('Humans', 282),
 ('Female', 128),
 ('Middle Aged', 89),
 ('Male', 85),
 ('Adult', 73),
 ('Aged', 67),
 ('Retrospective Studies', 41),
 ('Animals', 39),
 ('Neoplasms', 31),
 ('Treatment Outcome', 29)]
 ```
![2](https://user-images.githubusercontent.com/62388643/139727156-491258f6-2e91-4eb8-89d3-909fde26117e.png)
 
## Make a labeled table with rows for each of the top 5 MeSH terms from the Alzheimer's query and columns for each of the top 5 MeSH terms from the cancer query. For the values in the table, provide the count of papers (combined, from both sets) having both the matching MeSH terms. (5 points)
```python
l1 = ['Alzheimer Disease',
  'Humans',
 'Male',
 'Female',
 'Aged']
 
l2=['Humans',
 'Female',
 'Middle Aged',
 'Male',
 'Adult']

count={}
for i in l1:
    for j in l2:
        for art in ar:
            if i in ar[art]['mesh'] and j in ar[art]['mesh']:
                count[i+','+j] = count.get(i+','+j, 0)+1
                
count
```
```
>>>
{'Alzheimer Disease,Humans': 581,
 'Alzheimer Disease,Female': 244,
 'Alzheimer Disease,Middle Aged': 110,
 'Alzheimer Disease,Male': 290,
 'Alzheimer Disease,Adult': 26,
 'Humans,Humans': 982,
 'Humans,Female': 421,
 'Humans,Middle Aged': 236,
 'Humans,Male': 398,
 'Humans,Adult': 112,
 'Male,Humans': 398,
 'Male,Female': 353,
 'Male,Middle Aged': 202,
 'Male,Male': 445,
 'Male,Adult': 82,
 'Female,Humans': 421,
 'Female,Female': 435,
 'Female,Middle Aged': 217,
 'Female,Male': 353,
 'Female,Adult': 99,
 'Aged,Humans': 358,
 'Aged,Female': 314,
 'Aged,Middle Aged': 191,
 'Aged,Male': 305,
 'Aged,Adult': 67}
```

## Ideally, you can have the computer generate the table directly, but if not you could use nested for loops, label your output, and manually assemble a table in your readme document. Comment on any findings or limitations from the table and any ways you see to get a better understanding of how the various MeSH terms relate to each other. 
I generate a dataframe to show the table. We can see that the MeSH terms for the two fields are much overlapped, such as Humans, Male, Female. Even though there are terms more specifically like Alzheimer Disease, we can see there are still many papers companioned with the terms of Cancer due to the overlapping, so it is hard to know its relation to other terms and to distinguish the two fields only by the table or the MeSH terms.

```python
table = pd.DataFrame(data = np.array(np.random.randint(1,2,25)).reshape(5,5),columns = l1)
table.index=l2

for i in count:
    idx = i.split(',')
    table[idx[0]][idx[1]] = count[i]
table
```
```
>>>

  Alzheimer Disease	Humans	 Male    Female	  Aged
Humans	     581	982	  398	    421	   358
Female	     244	421	  353	    435	   314
Middle Aged  110	236	  202	    217	   191
Male	     290	398	  445	    353	   305
Adult	      26	112	  82	     99	    67
```

# EX 3
## In particular, for each paper identified from exercise 1, compute the SPECTER embedding (a 768-dimensional vector). Keep track of which papers came from searching for Alzheimers, which came from searching for cancer. 
The papers are processes in the json's keys' order(stored in a list), so the index of the matrix is 1-1 conrresponding to the keys, from which we can keep track of the queries.
```python
from transformers import AutoTokenizer, AutoModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

import json
with open('articles2019_n1.json', 'r', encoding='utf8') as f:
    papers = json.load(f)
    
keys = list(papers.keys())
```

```python
from tqdm import tqdm
import numpy as np
embeddings = []

batches = np.arange(0, len(keys), 2)
for batch in tqdm(batches):
    data = [papers[keys[i]]["ArticleTitle"] + tokenizer.sep_token + papers[keys[i]]["AbstractText"] for i in range(batch, min(batch+2, 1998))]
    inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)
    # take the first token in the batch as the embedding
    embeddings.append(result.last_hidden_state[:, 0, :].detach().numpy())
embeddings_= np.concatenate(embeddings)
```
## Apply principal component analysis (PCA) to identify the first three principal components
```python
from sklearn import decomposition
import pandas as pd
pca = decomposition.PCA(n_components=3)
embeddings_pca = pd.DataFrame(
    pca.fit_transform(embeddings_),
    columns=['PC0', 'PC1', 'PC2']
)
embeddings_pca["query"] = [paper["query"] for paper in papers.values()]
```
## Plot 2D scatter plots for PC0 vs PC1, PC0 vs PC2, and PC1 vs PC2; color code these by the search query used (Alzheimers vs cancer). 
```python
import matplotlib.pyplot as plt
plt.scatter(embeddings_pca[(embeddings_pca['query']=='Cancer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC0'], embeddings_pca[(embeddings_pca['query']=='Cancer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC1'], c='orange', alpha=.3, s=2.5, label = 'Cancer')
plt.scatter(embeddings_pca[(embeddings_pca['query']=='Alzheimer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC0'], embeddings_pca[(embeddings_pca['query']=='Alzheimer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC1'],c='navy', alpha=.3, s=2.5, label= 'Alzheimer')
plt.legend()
plt.title('PC0&PC1 by PCA')
plt.xlabel('PC0')
plt.ylabel('PC1')
plt.show()
```
![3](https://user-images.githubusercontent.com/62388643/139734098-ee827fd1-a77f-4535-b77a-1430284481d8.png)
```python
plt.scatter(embeddings_pca[(embeddings_pca['query']=='Cancer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC0'], embeddings_pca[(embeddings_pca['query']=='Cancer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC2'], c='orange', alpha=.3, s=2.5, label = 'Cancer')
plt.scatter(embeddings_pca[(embeddings_pca['query']=='Alzheimer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC0'], embeddings_pca[(embeddings_pca['query']=='Alzheimer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC2'],c='navy', alpha=.3, s=2.5, label= 'Alzheimer')
plt.legend()
plt.title('PC0&PC2 by PCA')
plt.xlabel('PC0')
plt.ylabel('PC2')
plt.show()
```
![4](https://user-images.githubusercontent.com/62388643/139734100-b7e48eaf-cd0f-478d-8c61-ea68f23b8c92.png)
```python
plt.scatter(embeddings_pca[(embeddings_pca['query']=='Cancer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC1'], embeddings_pca[(embeddings_pca['query']=='Cancer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC2'], c='orange', alpha=.3, s=2.5, label = 'Cancer')
plt.scatter(embeddings_pca[(embeddings_pca['query']=='Alzheimer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC1'], embeddings_pca[(embeddings_pca['query']=='Alzheimer')|(embeddings_pca['query']=='Alzheimer&Cancer')]['PC2'],c='navy', alpha=.3, s=2.5, label= 'Alzheimer')
plt.legend()
plt.title('PC1&PC2 by PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```
![5](https://user-images.githubusercontent.com/62388643/139734101-2c7fc021-e358-43b4-ace0-f78d44f0b871.png)
## Now look at the distribution with the LDA projection. Note that if you have n categories (presumably 2 or maybe 3 for you), LDA will give at most n-1 reduced dimensions... so graphically show the LDA projection results in the way that you feel best captures the distribution. Comment on your choice, things you didn't chose and why, and any other differences about what you saw with PCA vs LDA.
I have 3 classes ['Cancer', 'Alzheimer', 'Alzheimer&Cancer'], so there are at most 2 dims. I ploted both the distribution of 1d and 2d projections, and find that the 2d plot is more obvious to distinguish the 3 classes for a much more clean border.  
PCA selects the projection direction that accounts for the largest variance, and LDA selects the direction that maximizes the difference between groups and minimizes the variance within the group, which is supervised learning. So we can see the boarder between classes in LDA is much clearer than that of PCA. 
```pythhon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

categories = [paper["query"] for paper in papers.values()]
lda = LinearDiscriminantAnalysis(n_components=1)
embeddings_lda_1 = pd.DataFrame(
  lda.fit_transform(embeddings_, categories), columns=["lda0"]
)
lda = LinearDiscriminantAnalysis(n_components=2)
embeddings_lda_2 = pd.DataFrame(
  lda.fit_transform(embeddings_, categories), columns=["lda0","lda1"]
)
embeddings_lda_1["query"] = categories
embeddings_lda_2["query"] = categories
```
```python
plt.scatter([0]*len(embeddings_lda_1[embeddings_lda_1['query']=='Cancer']['lda0']),embeddings_lda_1[embeddings_lda_1['query']=='Cancer']['lda0'], c = 'orange',label='Cancer')
plt.scatter([0]*len(embeddings_lda_1[embeddings_lda_1['query']=='Alzheimer']['lda0']),embeddings_lda_1[embeddings_lda_1['query']=='Alzheimer']['lda0'], c = 'navy',label='Alzheimer')
plt.scatter([0]*len(embeddings_lda_1[embeddings_lda_1['query']=='Alzheimer&Cancer']['lda0']),embeddings_lda_1[embeddings_lda_1['query']=='Alzheimer&Cancer']['lda0'], c = 'red',label='Alzheimer&Cancer')
plt.title('1d projection by lda')
plt.legend()
plt.xticks([0], [''])
plt.ylabel('lda0')
plt.show()

```
![6](https://user-images.githubusercontent.com/62388643/139738769-a0a61ced-9628-4f0d-83f0-7c2eaf8c0339.png)

```python
plt.scatter(embeddings_lda_2[embeddings_lda_2['query']=='Cancer']['lda0'], embeddings_lda_2[embeddings_lda_2['query']=='Cancer']['lda1'], c='orange', alpha=.3, s=2.5, label = 'Cancer')
plt.scatter(embeddings_lda_2[embeddings_lda_2['query']=='Alzheimer']['lda0'], embeddings_lda_2[embeddings_lda_2['query']=='Alzheimer']['lda1'],c='navy', alpha=.3, s=2.5, label= 'Alzheimer')
plt.scatter(embeddings_lda_2[embeddings_pca['query']=='Alzheimer&Cancer']['lda0'], embeddings_lda_2[embeddings_pca['query']=='Alzheimer&Cancer']['lda1'],c='red', alpha=.3, s=2.5, label= 'Alzheimer&Cancer')
plt.legend()
plt.title('2d projection by lda')
plt.xlabel('lda0')
plt.ylabel('lda1')
plt.show()
```
![7](https://user-images.githubusercontent.com/62388643/139738771-8eb47347-e46f-40aa-b9d9-b6b82f647f0a.png)


# EX4
## Describe in words how you would parallelize this algorithm to work with two processes (5 points) and how you would validate the results and the speedup (5 points).
I would paritition the array into 2 equal parts, each thread processes 1 part due to the intrinsic partition recursion in the algorithm, and then merge the 2 ordered arrays. I would randomly generate arrays of different size(relatively large size to avoid the roundoff error), and go through some trials for each size(to balance randomness), to test the merge sort, paralleled merge sort and the broadly used  ```sorted(arr)``` method, whose underlying algorithm is a mix of quick sort and others, on their correctness and speed. 

## Use the multiprocessing module to implement a 2 process parallel version of the merge sort from slides3. Demonstrate that your solution can achieve at least a speedup of 1.5x. (5 points)
```python
import math
import multiprocessing
import random
import time
import numpy as np

def merge_sort(data):
    if len(data) <= 1:
        return data
    else:
        split = len(data) // 2
        left = iter(merge_sort(data[:split]))
        right = iter(merge_sort(data[split:]))
        result = []
        # note: this takes the top items off the left and right piles 
        left_top = next(left)
        right_top = next(right)
        while True:
            if left_top < right_top: 
                result.append(left_top) 
                try:
                    left_top = next(left)
                except StopIteration:
                    # nothing remains on the left; add the right + return
                    return result + [right_top] + list(right) 
            else:
                result.append(right_top) 
                try:
                    right_top = next(right)
                except StopIteration:
                    # nothing remains on the right; add the left + return
                    return result + [left_top] + list(left)
                    
                    
def merge(*args):
    left, right = args[0] if len(args) == 1 else args
    left_length, right_length = len(left), len(right)
    left_index, right_index = 0, 0
    merged = []
    while left_index < left_length and right_index < right_length:
        if left[left_index] <= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
    if left_index == left_length:
        merged.extend(right[right_index:])
    else:
        merged.extend(left[left_index:])
    return merged

def merge_sort_parallel(data):

    processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=processes)#4
    mid = len(data) // 2
    data = [data[:mid], data[mid:]]
    data = pool.map(merge_sort, data)
  
    data = merge(data[0], data[1])
    return data


if __name__ == "__main__":

    res = {'merge_sort': [], 'merge_sort_parallel': []}    
    for size in range(100000,1100000,100000):
        res['merge_sort'].append(0)
        res['merge_sort_parallel'].append(0)
        for trail in range(10):
            data_unsorted = [random.randint(0, size) for _ in range(size)]  
            
            for sort in merge_sort, merge_sort_parallel:
                
                start = time.time()
                data_sorted = sort(data_unsorted)
                end = time.time() - start
                res[sort.__name__][-1] += end
               
                assert sorted(data_unsorted) == data_sorted, 'merge sort not validated'
        
            res['merge_sort'][-1] /= 20
            res['merge_sort_parallel'][-1] /= 20


    print(res)
 ```
 ```
 >>> {'merge_sort_parallel': [0.03698297465552905, 0.06745909801219516, 0.11343476252494589, 0.21306876182170678, 0.19981711583980172, 0.2784967720520475, 0.2800033656297079, 0.29456480661852225, 0.34507608897790093, 0.4940746468024903], 'merge_sort': [0.06356818171759858, 0.11665013666115957, 0.17110447529951156, 0.26599080471600745, 0.33141187424245283, 0.46824584121189144, 0.4904059034067279, 0.5523171038771068, 0.6442017886016921, 0.7220174190322138]}
 ```
 
 Plot the time vs methods figure, which shows the time is shinked to approximately 2/3. Calcualting the mean of its fraction, we get 1.6x.
 ```python
import matplotlib.pyplot as plt
plt.plot( range(10000,110000,10000), res['merge_sort_parallel'], c = 'orange')
plt.plot( range(10000,110000,10000), res['merge_sort'], c = 'green')
plt.xlabel('array size')
plt.ylabel('time')
plt.title('speed of merge sort vs paralleled merge sort')
plt.show()
```
![9](https://user-images.githubusercontent.com/62388643/139760422-5b0455c1-681d-4b26-b271-74b8c846a58c.png)

 
 ```python
import numpy as np
np.mean([res['merge_sort'][i]/res['merge_sort_parallel'][i] for i  in range(10)])
```
```
>>> 1.649938131351138
```
# EX5


## Do data exploration on the dataset you identified in exercise 4 in the last homework, and present a representative set of figures that gives insight into the data. Comment on the insights gained. (5 points)
There are 5 predictors and 1 target pressure. 

R - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change R by changing the diameter of the straw, with higher R being harder to blow.
C - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change C by changing the thickness of the balloon’s latex, with higher C having thinner latex and easier to blow.
time_step - the actual time stamp.
u_in - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
u_out - the control input for the exploratory solenoid valve. Either 0 or 1.
pressure - the airway pressure measured in the respiratory circuit, measured in cmH2O.
Breath_id is globally-unique time step for breaths. And each step has 80 samples.
```python
train.groupby('breath_id')['id'].count()
```
```
breath_id
1         80
2         80
3         80
4         80
5         80
          ..
125740    80
125742    80
125743    80
125745    80
125749    80
Name: id, Length: 75450, dtype: int64
```
U_out is the control input for the exploratory solenoid valve. It is either 0 or 1.
```python
train['u_out'].value_counts()
```
```
1    3745032
0    2290968
Name: u_out, dtype: int64
```
The remains statistical summary is as follows.
```
train[['R','C','u_in','pressure']].describe()
```
```

             R	           C	           u_in	         pressure
count	6.036000e+06	6.036000e+06	6.036000e+06	6.036000e+06
mean	2.703618e+01	2.608072e+01	7.321615e+00	1.122041e+01
std	1.959549e+01	1.715231e+01	1.343470e+01	8.109703e+00
min	5.000000e+00	1.000000e+01	0.000000e+00	-1.895744e+00
25%	5.000000e+00	1.000000e+01	3.936623e-01	6.329607e+00
50%	2.000000e+01	2.000000e+01	4.386146e+00	7.032628e+00
75%	5.000000e+01	5.000000e+01	4.983895e+00	1.364103e+01
max	5.000000e+01	5.000000e+01	1.000000e+02	6.482099e+01
```
Distribution of pressure.
```python
plt.violinplot(train['pressure'])
plt.ylabel('pressure')
plt.title('Distribution of pressure')
plt.show()
```
![8](https://user-images.githubusercontent.com/62388643/139754425-4f3cee6e-09d3-4296-9713-b98895f0f2da.png)

Distribution of u_in.
```python
plt.hist(train['u_in'],bins=20)
plt.title('Distribution of u_in')
plt.xlabel('u_in')
plt.show()
```
![99](https://user-images.githubusercontent.com/62388643/139754427-32d8ce62-f7db-41f1-99b4-9eb7e8c49ce7.png)

Pivot table with each breath id and its samples. R*/C is constant.
```python
pt = pd.pivot_table(train, index=['breath_id','RC','u_in','u_out'], values=['pressure'])
```
```

                                       pressure
breath_id RC	     u_in	u_out	
1	 1000	0.000000	 1 	9.240116
                0.083334	0	5.837492
                0.779225	1	7.524743
                1.439041	1	6.962326
                1.994220	1	7.454441
...	...	...	...	...
125749	 500	14.700098	0	17.015533
                16.101221	0	15.468886
                16.266744	0	15.117375
                21.614707	0	9.563505
                25.504196	0	5.345377
```
Correlation map of those varibles for each step.
```python
import seaborn as sns
sns.heatmap(train[train['breath_id']==1][['u_in','u_out','pressure']].corr(), annot=True)
plt.title('Correlation map')
plt.show()
```
![10](https://user-images.githubusercontent.com/62388643/139754426-b1ed80c1-1403-49a7-b8be-5038d3e68537.png)




## Are there any missing values in your data? (Whether the answer is yes or no, include in your answer the code you used to come to this conclusion.) If so, quantify. Do they appear to be MAR/MCAR/MNAR? Explain. (5 points) <-- This will be discussed in class on October 12.
No. Check with df.isnull() as follows. Also, the description table and distribution plot above also show a reasonable distribution without any 0s or -1s.
```python
df = pd.read_csv("ventilator-pressure-prediction/train.csv", na_values = ['--', 'N/A', 'na'])
print(df.isnull().sum())
```
```
id           0
breath_id    0
R            0
C            0
time_step    0
u_in         0
u_out        0
pressure     0
dtype: int64
```
## Identify any data cleaning needs and write code to perform them. If the data does not need to be cleaned, explain how you reached this conclusion. (5 points)


There is no cleaning need. According to above data desciption, all data is standardized in the uniform formatting, and is tidying data. Also, per the above analysis, there is not any missing data or duplicates.

```python
len(train['id'].unique())==len(train)
```
```
>>> True
```

