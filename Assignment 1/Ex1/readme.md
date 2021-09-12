# Ex1 @Weijin ZOU
All code blocks and presented results can be tested and viewed in [```HW1-Ex1.ipynb```](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%201/Ex1/HW1-Ex1.ipynb)

## Write a function temp_tester that takes a definition of normal body temperature and returns a function that returns True if its argument is within 1 degree of normal temperature

I write a class in which you could initial a test for whatever item by just setting its normal temprature, and call this function just by passing the testing temperature after initializing it.

```python
# define a temp_test class
class temp_tester(object):
    def __init__(self, norm):
        self.norm = norm
    def __call__(self, test):
        if abs(test-self.norm) > 1:
            return False
        else:
            return True
```
## Test code with the following 
```python
# test code
human_tester = temp_tester(37)
chicken_tester = temp_tester(41.1)
```
```python
chicken_tester(42)
```
```python
>>> True
```
```python
human_tester(42)
```
```python
>>> False
```
```python
chicken_tester(43)
```
```python
>>> False
```
```python
human_tester(35)
```
```python
>>> False
```
```python
human_tester(98.6)
```
```python
>>> False
```
