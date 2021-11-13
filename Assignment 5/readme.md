# EX1
## implement a two-dimensional version of the gradient descent algorithm to find optimal choices of a and b. (7 points) 
## Explain how you estimate the gradient given that you cannot directly compute the derivative (3 points), identify any numerical choices -- including but not limited to stopping criteria -- you made (3 points), and justify why you think they were reasonable choices (3 points).
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
# EX4
Implement a function that takes two strings and returns an optimal local alignment (6 points) and score (6 points) using the Smith-Waterman algorithm; insert "-" as needed to indicate a gap (this is part of the alignment points). Your function should also take and correctly use three keyword arguments with defaults as follows: match=1, gap_penalty=1, mismatch_penalty=1 (6 points). Here, that is a penalty of one will be applied to match scores for each missing or changed letter.
Test it, and explain how your tests show that your function works. Be sure to test other values of match, gap_penalty, and mismatch_penalty (7 points).
