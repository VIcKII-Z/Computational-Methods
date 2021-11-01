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

Alzheimer Disease	Humans	Male	Female	Aged
Humans	581	982	398	421	358
Female	244	421	353	435	314
Middle Aged	110	236	202	217	191
Male	290	398	445	353	305
Adult	26	112	82	99	67
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


