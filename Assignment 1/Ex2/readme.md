# Ex2 @Weijin ZOU

All code blocks and presented results can be tested and viewed in [```HW1-Ex1.ipynb```]()

## Examine data. What columns does it have?  How many rows (think: people) does it have?

List the columns:
```python
list(data.columns)
```
```python
>>> ['name', 'age', 'weight', 'eyecolor']
```
The number of rows:
```python
data.shape[0]
```
```
>>> 152361
```
## Examine the distribution of the ages in the dataset. In particular, be sure to have your code report the mean, standard deviation, minimum, maximum. Plot a histogram of the distribution with an appropriate number of bins for the size of the dataset 

Use df.describe() to summarize the data properties:

```python
data.describe()
```
```
        age	        weight
count	152361.000000	152361.000000
mean	39.510528	60.884134
std	24.152760	18.411824
min	0.000748	3.382084
25%	19.296458	58.300135
50%	38.468955	68.000000
75%	57.623245	71.529860
max	99.991547	100.435793
```

Plot the histogram of the distribution, and here I choose the number of bins to be 20. If smaller, the histogram doesn't really portray the data very well. And if even larger, the histogram will have a broken comb look, which also doesn't give a sense of the distribution. Also, the range 100 could be divided by 20, which makes 20 sounds good too.
```python
plt.hist(data['age'],bins=20, rwidth=.9)
plt.title('Frequency distribution of ages')
plt.xlabel('age')
plt.show()
```
![](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%201/Ex2/figure%201.png)

As the figure shows, the ages of most participants are between 0 and 65 and are distributed evenly. Also there were also a few people older than 65, and the ages of these people are distributed evenly as well.

## Repeat the above for the distribution of weights. 
Here I also choose 20 for its 
```python
plt.hist(data['weight'],bins=20, rwidth=.9)
plt.title('Frequency distribution of weights')
plt.xlabel('weight')
plt.show()
```
![](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%201/Ex2/figure%202.png)

For the distribution of weights, it is more like a normal distribution around the mean of 70.

## Make a scatterplot of the weights vs the ages. Describe the general relationship between the two variables.

```python
x = data['age']
y = data['weight']
plt.scatter(x, y, s=3)
plt.xlabel('age')
plt.ylabel('weight')
plt.show()

```
![](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%201/Ex2/figure3.png)

As the fighre shows, generally when the age is under 25, the weights increase linearly as ages increase, and then stay relatively stable after the age is above 25.



## You should notice at least one outlier that does not follow the general relationship. What is the name of the person?  Be sure to explain your process for identifying the person whose values don't follow the usual relationship in the readme. 


From above figure, there is one person whose age is above 40, yet his weight is under 25, which is unique in those who are above 40. I therefore set these two conditions as filtering criteria, and finally found this person.

```python
data[(data['age']>40)&(data['weight']<25)]
```
```
	name	        age	weight   eyecolor
537	Anthony Freeman	41.3	21.7	 green
```
