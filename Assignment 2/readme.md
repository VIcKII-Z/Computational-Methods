# EX1 @WeijinZOU
All code blocks and presented results can be tested and viewed in [```HW2-Ex1.ipynb```](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%202/Ex%201/HW2_EX1.ipynb)

## Load the data. Plot a histogram showing the distribution of ages (2 points). Do any of the patients share the same exact age? (2 points) How do you know? (2 points)

Use cElementTree to parse xml data, and track missing data by ```assert (len(name)==len(age) and len(age) == len(gender))```.

```python
import xml.etree.cElementTree as ET
tree = ET.parse('hw2-patients.xml')
root = tree.getroot()

for patient in root.getchildren()[-1].findall('patient'):
    name.append(patient.attrib['name'])
    age.append(patient.attrib['age'])
    gender.append(patient.attrib['gender'])
    assert (len(name)==len(age) and len(age) == len(gender)), 'miss things'
```
Load it to pd.DataFrame:
```python
import pandas as pd
patient_tb = pd.DataFrame({'name':name, 'age':pd.to_numeric(age), 'gender': gender})
patient_tb
```
```
name	age	gender
0	Tammy Martin	19.529988	female
1	Lucy Stribley	1.602197	female
2	Albert Trevino	19.317023	male
3	Troy Armour	79.441208	male
4	Jose Masseria	71.203863	male
...	...	...	...
324352	Jeremy Brode	60.955355	male
324353	Lynda Brown	22.676277	female
324354	Joyce Adkins	64.466378	female
324355	Kevin Hensley	56.770128	male
324356	James Hawk	69.372302	male
```


Plot a histogram:
  
```python
plt.hist(patient_tb['age'], rwidth = 0.9)
plt.title('Distribution of ages')
plt.xlabel('age')
plt.show()
```
   
![](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%202/Ex%201/figure%201.png)

Check if some patients share the same age:
```python
len(patient_tb['age'].unique())==len(patient_tb['age'])
```
```
>>> True
```
So the answer is NO.

## For an extra 2 points: explain how the answer to the question about multiple patients having the same age affects the solution to the rest of the problem.

If there are patients sharing the same age, we could not step out of the while loop in the bisection function once we find only one target number. Since the point is no longer to find the number equals the target, but the number which has no numbers larger than it to its left or no smaller number to its right, which, indeed is its new stopping criterion. 

## Plot the distribution of genders. (2 points). In particular, how did this provider encode gender? What categories did they use? (2 points)
Plot the distribution of genders:

![](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%202/Ex%201/figure%202.png)  


This provider encode gender as female, male and unknown.

## Sort the patients by age and store the result in a list (use the "sorted" function with the appropriate key, or implement sort yourself by modifying one of the algorithms from the slides or in some other way). (2 points) Who is the oldest patient? (2 points).
Sort the age and store it in a list.
```python
sorted_age = patient_tb.sort_values(by = 'age', ascending=False)
age_list = list(sorted_age['age'])
```
Check the first item in the sorted age dataframe to get the oldest one, who is Monica Caponera, and she is 84.998557 years old.
```python
sorted_age.iloc[0]
```
```
name      Monica Caponera
age             84.998557
gender             female
Name: 124560, dtype: object
```
## Identifying the oldest person from a list sorted by age should be an O(1) task... but sorting is an O(n log n) process (assuming we're using an efficient algorithm), so the total time for the above is O(n log n). Describe how (you don't need to implement this, unless that's easier than writing it out) you could find the second oldest person's name in O(n) time. (2 points). Discuss when it might be advantageous to sort and when it is better to just use the O(n) solution. (2 points).

Use two variables max and snd_max to store the ages, and iterates through the list, and the total time is O[N].
```python
max, snd_max = 0, 0
for i in patient_tb['age']:
    if i > max:
        max, snd_max = i, max
        
    elif i > snd_max and i < max:
        snd_max = i
print(max, snd_max)
```
```
84.99855742449432 84.9982928781625
```
```python
patient_tb[patient_tb['age']==84.9982928781625]
```
```
	name	age	gender
253020	Raymond Leigh	84.998293	male
```
Here the second oldest person is Raymond Leigh, who is 84.9982928781625 years old. When N is not so large and therefore NlogN is not so distinctly different from N, or, sorting benefits the downstream pipeline, it is better to sort. Otherwise, if only focusing on who is the second, O(n) method is more efficient.

## Recall from our discussion of the motivating problem for September 9th that we can search within a sorted list in O(log n) time via bisection. Use bisection on your sorted list (implement this yourself; don't trivialize the problem by using Python's bisect module) to identify the patient who is 41.5 years old. (2 points)
Use bisection:
```python
l, r = 0, len(age_list)
while l <= r :
    mid = int(l + (r-l)/2)
    if age_list[mid] == 41.5:
        break
    elif age_list[mid] < 41.5:
        r = mid - 1
    else:
        l = mid + 1
print(sorted_age.iloc[mid])
```
```
name      John Braswell
age                41.5
gender             male
Name: 939, dtype: object
```
As the above shows, the pateint who is 41.5 years old is John Braswell.

## Once you have identified the above, use arithmetic to find the number of patients who are at least 41.5 years old. (2 points)
```python
mid+1
```
```
>>> 150471
```
150471 patients are at least 41.5 years old.


## Generalizing the above, write a function that in O(log n) time returns the number of patients who are at least low_age years old but are strictly less than high_age years old. (2 points) Test this function (show your tests) and convince me that this function works. (2 points). (A suggestion: sometimes when you're writing high efficiency algorithms, it helps to make a slower, more obviously correct implementation to compare with for your tests. Be sure your function works both for ages that are and are not in the dataset.)


Use bisection ```BinarySearch``` two times for the high bound and low bound of ages. Also, construct a function ```N_search_methd``` with a slower but obviously correct method, which is to itertate throught the list with O(N) time.    

Finally, construct a random sample generater ```randomGenerater```, which is used for 1000 trials(could be larger if you set another number) to test if the two methods reach the same result.  

Also, it is noteworthy that ```value check``` is used for the boundary treatment of the binary search.  

```python
def BinarySearch(arr, target):
    l, r = 0, len(arr)
    while l <= r :
        mid = int(l + (r-l)/2)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            r = mid - 1
        else:
            l = mid + 1
    return mid
def age_scale_idf(age_list, high_age, low_age):
    """
    param: age_arr  list of sorted ages
    """
    # value check
    if low_age < np.min(age_list):
        low_age = np.min(age_list)
    if high_age <= low_age:
        raise ValueError('low should be smaller than high')
    
    # binary search
    high_bound = BinarySearch(age_list, high_age)
    low_bound = BinarySearch(age_list, low_age)
    
    if age_list[high_bound] >= high_age:
        high_bound +=1
    if age_list[low_bound] < low_age:
        low_bound -=1
    return low_bound-high_bound+1

def N_search_methd(age_list, high_age, low_age):
    
    # value check
    if high_age <= low_age:
        raise ValueError('low should be smaller than high')
    
    count = 0
    for i in age_list:
        if low_age <= i < high_age:
            count+=1
        elif i <  low_age:
            break
   
    return count
```
Use one simple sample to check:
```python
age_scale_idf(age_list, 40, 20)
```
```
>>> 85524
```
Case of boundary overflow:
```python
age_scale_idf(age_list, 100, 0)
```
```
>>> 324357
```

Then use O(N) method to compare with the testing method, and conduct 1000 trials.

```python
        
for trial in range(10**3):
    random_Generator = 84*np.random.random(2)
    high_age, low_age = np.max(random_Generator), np.min(random_Generator)
    assert age_scale_idf(age_list, high_age, low_age) == N_search_methd(age_list, high_age, low_age), 'test failed'

print('1000 tests success')

```
```
>>> 1000 tests success
```
As the above shows, the tester with the random sample generator goes through all 1000 trials, and those look all fine. You could set the trial number to a even larger one like 1e8.

## Modify the above, including possibly the data structure you're using, to provide a function that returns both the total number of patients in an age range AND the number of males in the age range, all in O(log n) time as measured after any initial data setup. (2 points). Test it (show your tests) and justify that your algorithm works. (2 points)

Devide the DataFrame to two according to the gender, whether is male or not. Call age range identification with bisection function ```age_scale_idf``` twice and return the number in this range respectively. Then we get total number in this range and also the number of male in this range.

Also, based on the random sample generator, test the algorithm with O(N) methods for multiple times.


```python
def age_and_gender(data, low_age, high_age):
    # value check
    if low_age >= high_age:
        raise ValueError('low should be smaller than high')
    
    data_notmale, data_male = data[data['gender']!='male'], data[data['gender'] == 'male']
    num_notmale, num_male = age_scale_idf(list(data_notmale['age']), high_age, low_age), age_scale_idf(list(data_male['age']), high_age, low_age)
    
    # algorithm check
    assert num_notmale == N_search_methd(list(data_notmale['age']), high_age, low_age) and num_male == N_search_methd(list(data_male['age']), high_age, low_age), 'test failed'
    
    return f'Total number of age range between {low_age} and {high_age} is {num_notmale+num_male}, and {num_male} males are in this range. '
    
    
```
Test it with small part of the data.
```python
age_and_gender(sorted_age[:10], 41.5, 100)
```
```
>>> 'Total number of age range between 41.5 and 100 is 10, and 4 males are in this range. '
```

Test it with one sample with the whole data:
```python
age_and_gender(sorted_age, 41.5, 100)
```
```
>>> 'Total number of age range between 41.5 and 100 is 150471, and 71308 males are in this range. '
```

Also, test it for multiple times based on random samples.
```python
for trial in range(10**3):
    random_Generator = 100*np.random.random(2)
    high_age, low_age = np.max(random_Generator), np.min(random_Generator)
    age_and_gender(sorted_age, low_age, high_age)

print('1000 tests success')
```
```
>>> 1000 tests success
```
Still, the result shows the function works well. 

# EX2 
All code blocks and presented results can be tested and viewed in [```HW2-Ex2.ipynb```](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%202/Ex%202/HW2_EX2_.ipynb)

## Write code to loop through all 15-mers, subsequences of 15 bases within chromosome 2 (CM000664.2)
```python
for chromosome in human_genome:
    if chromosome.name == "CM000664.2":
        sequence = str(chromosome.seq).lower().encode('utf8')
        for i in range(len(sequence) - 15):
            mers_15 = sequence[i : i + 15]
            
```
## How many total subsequences are there (counting duplicates) that do not contain more than 2 Ns? 
```python
for chromosome in human_genome:
    if chromosome.name == "CM000664.2":
        sequence = str(chromosome.seq).lower()
        # DO STUFF HERE
        count = 0
        for i in range(len(sequence) - 15):
            subseq = sequence[i : i + 15]
            count_inseq = subseq.count("n") 
            if count_inseq <= 2:
                count += 1
        print(count)
 ```
 ```
 >>> 240548031
 ```
## Using 100 hash functions from the family below and a single pass through the sequences, estimate the number of distinct 15-mers in the reference genome's chromosome 2 using the big data method for estimating distinct counts discussed in class. (5 points) 

To save the memmory and upgrade the speed, I did the calculation within the data stream  with O(1) extra space. And the total time is limited to 8h.
```python
min_hashes = [scale]*100
for chromosome in human_genome:
    if chromosome.name == "CM000664.2":
        sequence = str(chromosome.seq).lower().encode('utf8')
        for i in range(len(sequence) - 15):
            mers_15 = sequence[i : i + 15]            
            if mers_15.count('n'.encode('utf8')) < 3:
                for j in range(100):
                    f = hash_funcs[j]
                    tmp = f(mers_15)
                    min_hashes[j] = min(min_hashes[j], tmp)
	    # Progress monitor
            if i%1000000 == 0:
                print(f'---{i+1}th  ---'+time.asctime()+':',min_hashes)
                
        break     
num = int(scale/np.median(min_hashes) -1)
print(f'Number of estimated distinct mers is {num}')
                
```
```
>>> Number of estimated distinct mers is 201523391.
```
The esimated distinct mers number is 201523391.
## How does your estimate change for different-sized subsets of these hash functions, e.g. the one with a=1 only, or a=1, 2, .., 10, or a=1, 2, ...100, etc? (5 points) 
Using the min_hashes list generated above, the reason I can do this is that when a(hash function) and the data are fixed, then the min value of hashes is fixed. When a = 1, estimated number of distinctive 15-mers is 66076418. When a = 1, 2, ..., 10, estimated number of distinctive 15-mers is 138827224. And when a = 1,..100 as calculated above, is 201523391. As the number of hash functions grows, the error caused by hash collision reduces, so the number of distinctive rises.
```python
res1 = int(scale / min_hashes[0] - 1)
res10 = int(scale / np.median(min.hashes[:10]) - 1)
print(res1)
print(res10)
```
```
>>> 66076418
>>> 138827224
```

## Explain your tests and why they convinced you that your code works

I generated a fake data consisting of 1000000 strs and 10 of those are distinct to test my algorithm. And the result shows the estimate is quite near to the true value.
```python
import random
import numpy as np

def str_generator(n):
    s = []
    for i in range(n):
        s.append(''.join(random.sample(['z','y','x','w','v','u','t','s','r','q','p','o','n','m','l','k','j','i','h','g','f','e','d','c','b','a'], 20)).encode('utf8'))
    return s
my_mers_list = str_generator(10)*100000
print("Number of total mers :", len(my_mers_list))
print("Number of unique mers :", len(set(my_mers_list)))

my_min_hashes = []
for k in range(100):
    my_hash_func = hash_funcs[k]
    my_min_hash = float('inf')
    for subseq in my_mers_list:
        temp = my_hash_func(subseq)
        my_min_hash = min(my_min_hash, temp)
    my_min_hashes.append(my_min_hash)
num = scale/np.median(my_min_hashes) - 1
print("Number of estimated unique mers :", num) 
```
```
>>> Number of total mers : 1000000
>>> Number of unique mers : 10
>>> Number of estimated unique mers : 11.798358161891802
```

# EX3

## Explain what went wrong (6 points). 
There may be several reasons causing the problem:  
Fisrt, he may use a 32-bit python instead of a 64-bit one. The 32-bit operating system can address the memory range of 2^32, and the 64 bit operating system can address memory range of 2^64. In other words, a 32-bit operating system can only use about 4GB of memory in theory. If a program wants to use more than 4GB of memory, it must choose a 64 bit operating system.  
Second, even if he uses a 64-bit system, this problem may also occur. This is because some overhead or background activity on his computer maybe taking up RAM  
Third,on some operating systems, there are limits to how much RAM a single CPU can handle. So even if there is enough RAM free, a single thread cannot take more.  
Last but not least, the algorithm he picks. In some cases like this, he does not need a list to store all of the data, but a generator func or iterator is enough.



## Suggest a way of storing all the data in memory that would work (7 points)
For large datasets, instead of loading the entire dataset into memory, he could keep the data in the hard drive and access it in batches. For example, divide the file into pieces and load/deal with the data seperately. Or use some libraries to get trunks of the data and deal with it by trunks, like pd.get_trunk(). 
## suggest a strategy for calculating the average that would not require storing all the data in memory (7 points).
Use two variables to record the number and sum could give the average instead of calculating after storing all data. The code is as follows.
```python
with open('weights.txt') as f:
    s = 0
    count = 0
    for line in f:
        s += float(line)
        count += 1
print("average =", s / count)
```


# EX4 
All code blocks and presented results can be tested and viewed in [```HW2-Ex4.ipynb```](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%202/Ex%204/HW2-EX4.ipynb)
## Identify a data set online (10 points) that you find interesting that could potentially be used for the final project; 
I use data from Kaggle competition: https://www.kaggle.com/c/ventilator-pressure-prediction.  
Using this data, I could simulate a ventilator connected to a sedated patient's lung taking lung attributes compliance and resistance into account.
I am interested in this for it could help overcome the cost barrier of developing new methods for controlling mechanical ventilators. 
This will pave the way for algorithms that adapt to patients and reduce the burden on clinicians during these novel times and beyond. 
As a result, ventilator treatments may become more widely available to help patients breathe.

## Describe the dataset (10 points) 
There are 2 files. One for training and the other one for testing.
```
train.csv - the training set
test.csv - the test set
```
* Varibales:
There are 8 variables, whose detailed descriptions are as below.
```
id - globally-unique time step identifier across an entire file  
breath_id - globally-unique time step for breaths  
R - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change R by changing the diameter of the straw, with higher R being harder to blow.  
C - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change C by changing the thickness of the balloon’s latex, with higher C having thinner latex and easier to blow.  
time_step - the actual time stamp.  
u_in - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.  
u_out - the control input for the exploratory solenoid valve. Either 0 or 1.  
pressure - the airway pressure measured in the respiratory circuit, measured in cmH2O.  
```
* Are the key variables explicitly specified or are they things you would have to derive (e.g. by inferring from text)? 
The key variables are explicitly specified.
 

* Are any of the variables exactly derivable from other variables? (i.e. are any of them redundant?) 
For now I think none of these are reudundant. But it would denpend on the follow-up exploration.

* Are there any variables that could in principle be statistically predicted from other variables? 
Yes. Given numerous time series of breaths and the airway pressure in the respiratory circuit during the breath should be predicted, given the time series of control inputs.
* How many rows/data points are there? 

```
train.shape, test.shape
```
```
((6036000, 8), (4024000, 7))
```
For training data, there are 6036000 rows, and for testing data, there are 4024000 rows. 
* Is the data in a standard format? If not, how could you convert it to a standard format?
Yes, it is. Below is its format.
```
id	breath_id	R	C	time_step	u_in	u_out	pressure
0	1	1	20	50	0.000000	0.083334	0	5.837492
1	2	1	20	50	0.033652	18.383041	0	5.907794
2	3	1	20	50	0.067514	22.509278	0	7.876254
3	4	1	20	50	0.101542	22.808822	0	11.742872
4	5	1	20	50	0.135756	25.355850	0	12.234987
```

## Describe the terms of use and identify any key restrictions (e.g. do you have to officially apply to get access to the data? Are there certain types of analyses you can't do?) (5 points)
Below is the rules of data access and use clarified by the host. The data is open source and could be used for academic research and so on, but only for non-conmercial purposes.
>Data Access and Use. You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.
