# Ex4 @Weijin ZOU

All code blocks and presented results can be tested and viewed in [```HW1-Ex4.ipynb```](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%201/Ex4/HW1-Ex4.ipynb)

## Make a function that takes a population size n and a number of drug users d and returns a list of size n with d True values and n - d False values. The Trues represent drug users, the Falses represent non-drug users. 

In this code I made anomaly detection in case n<d.

```python
def return_values(n, d):
    if d<=n:
        return d*[True]+(n-d)*[False]
    else:
        raise ValueError('n<d')
```
```python
# test code
return_values(5,3)
```
```python
>>> [True, True, True, False, False]
```
```python
return_values(3,5)
```
```python
>>> ValueError: n<d
```

## Make a function that selects a sample of size s (think: study participants) from such a population and returns that sample's responses to the protocol from the slides. 

I firstly construct a random_pick function to simulate a random process. It takes the events list and each event's probability, and then returns the output of one stochastic simulation.

```python
import random
def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability: 
            break
    return item
```
Then I made the function that selects sample of size k from a population of n, and return the sample's response list sp_k, which includes False(NO) and True(YES).
```python
def report(n, k, p):
    """
    :param n: population
    :param k: sample number
    :param p: fraction of drug users
    :return: sample's responses list 
    
    """
    omega = [i for i in range(n)]
    sp_k = random.sample(omega, k)
    for i in range(k):
        tmp = random.randint(0,1)
        if tmp ==0:
            sp_k[i] = random_pick([False, True], [0.5, 0.5])
        else:
            sp_k[i] = random_pick([False, True], [1-p, p])
    return sp_k
 ```
 ## Make a function that takes parameters for the total population, the true number of drug users in the population, the sample size, and returns the estimated number of drug users in the population that would be predicted using such a sample and the randomized response protocol. 

For ease of later use, I add param __rpt__ representing the number of repeated experiment, which is now simply set to be default value 1. Also, I use ratio of drug users instead of exact number to avoid annoying calculation in loops in later use. Currently, we could just multiple it with sample size.

```python
def predict(n, d, k, rpt=1):
    prob = 0   
    # repeat rpt times
    for i in range(rpt):
        sp_k = report(n, k, d/n)
        # if the predicted rate is less than 0, assume 0.
        prob += max(2*sp_k.count(True)/k-0.5, 0)    
    prob /= rpt
#     usr_num = int(prob * k)
    return prob
```
```python
# test code
int(predict(1000,100,50)*50)
```
```
>>> 13
```
## Analyze the going negative issue in some way that is interesting to you and provide a brief written discussion and either a figure or a statistical analysis.

It's worth noting that I add ```prob += max(2*sp_k.count(True)/k-0.5, 0) ``` in my code to avoid the negetive issues. But why this issue happens?
We could do calculation about when p goes negative base on E[yes] = 0.25 + 0.5p. The result shows p<0 if  c < k/4, where c is the count of yes in sample, k is the sample size. Furthermore, c relys much on d/n. When d/n\<\<k, therefore, p has a relatively large possibility to be negative. Below is an vivid example when d/n\<\<k, and the predicted ratio to be negative.  

_Notice that to test the negative issues, the code in above block should be : prob += 2*sp_k.count(True)/k-0.5_ 
```python
predict(10000000,1,10)
```
```
>>> -0.09999999999999998
```


## Suppose that we have a population of 1000 people, 100 of whom are drug users and we do a survey using this protocol that samples 50 people from the total population. What is the estimated number of drug users you get from such an approach? 


```python
# test code
int(predict(1000,100,50)*50)
```
```
>>> 13
```

## Your results in part d will obviously depend on which 50 people you surveyed and how their coin-flips worked out. To get a sense of the distribution, repeat the experiment many times.

By change the parameter __rpt__ in the above function __predict__, we could simulate the experiemnt for rpt times. And to decide on the times of repeating, I set  the standards that the result is considered to be stable when the difference between two experiments is less than __0.01__(_ie. the diff of the ratio of drug users of two successive experiments_) during the last __20__ experiments.

Also, to eliminate the calculating, I add __rpt__ to present the initial considered repeat number, which means we do not need to begin with 1 experiment but a relatively larger number closed to the number we expected. 

```python
def repeat_exp(n, d, k, rpt=1):
    flag  = 1
    usr_nums = [predict(n, d, k, rpt) * k]
    while flag>0:
        new, prev = predict(n, d, k, rpt+1), predict(n, d, k, rpt)       
        diff = abs(new - prev)
        if diff < 0.01:
            flag -= 0.05
        else:
            flag = 1
        rpt += 1
        usr_nums.append(new * k)
    plt.hist(usr_nums, bins = 20)
    plt.show()    
    return rpt
```
Test the above func using (1000,100,50) example, and the result goes to 576, which means to conduct 576 experiments is probably enough for the prediction of the true drug user rate.   
Notice that the result is not constant for the intrinsic randomness.
```python
# test code
repeat_exp(1000,100,50,100)
```
```
>>> 576
```
### Plot a histogram showing the predictions 

Conduct the experiment for ```rpt``` times, here 576 given by the above function, and plot the histogram showing preditions.

```python
def plot_distrubution(n,d,k,rpt):
    usr_nums = []
    for i in range(1,rpt): 
        usr_nums.append(int(predict(n, d, k, i) * k))
            
    plt.hist(usr_nums, bins = 20)
    plt.title(f'Frequency distribution of drug users in {rpt} repeated experiments')
    plt.show()
```
```python
plot_distrubution(1000,100,50,576) 
```
![](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%201/Ex4/figure%201.png)

## Repeat parts d and e but with a population of 100_000 people, 10_000 drug users and sampling 5_000 people; i.e. with everything scaled up by a factor of 100. How do your results compare? 

Use __repeat_exp__ to get the proper repeat times.

```python
repeat_exp(100000,10000,5000)
```
```
>>> 30
```
Then plot the distibution. 
```python
plot_distrubution(100000,10000,5000,30) 
```
![](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%201/Ex4/figure%202.png)

We can see that most results of the experiments are near 500, which is correct due to the true drug user ratio(10000/100000 = 0.1).  


Compared with that of last example, the proper repeat times is much smaller. The reason why this happens is that although the ratio of drug users is the same as last example(0.1), the sample size grows 100 times bigger, which, to a large extent, **reduces the sampling error and improves the accuracy of simulations. In other words, with the increase of the number of samples, the sample structure is closer to the population.**
Thus, in this example we need much less experiments to reach the stable predition result.


## Repeat parts d and e but with 500 drug users in a population of 1_000 and sampling 50 people. i.e. with the smaller population but with higher drug usage rates.  How do your results compare?

```python
repeat_exp(10000,500,50,100)
```
```
>>> 357
```
```python
plot_distrubution(1000,500,50,357) 
```
![](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%201/Ex4/figure%203.png)

Also, we compare the result with the fist example and find the ratio of drug users increasing from 0.1 to 0.5. **When other conditions remain unchanged, the smaller the variation, the smaller the sampling error. This is because small variation of population leads to small variation in sample. More extremely, when the variation of the population is 0, say all non-drug-users, the sample's properties are equal to population's, and thus no sampling error occurs.**



