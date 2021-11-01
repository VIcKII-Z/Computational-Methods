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
I simply use string concatenation. I frist tried to use dict for I notice I could strore the abstract corresponding to their labels, such as {'BACKGROUND':xxx,...}. 
However, I found the labels do not count much to our analysis in this case, so I choose to concatenate the string. The strength is that it is super convienient for text analysis thansk to its simple formatting as strings. It also has weakness like the memory cost and infexibility.
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
