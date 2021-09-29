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


This provider encode gender as female, male and unkown.

## Sort the patients by age and store the result in a list (use the "sorted" function with the appropriate key, or implement sort yourself by modifying one of the algorithms from the slides or in some other way). (2 points) Who is the oldest patient? (2 points).S
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
Here the second oldest person is Raymond Leigh, who is 84.9982928781625 years old. When N is not so large and therefore NlogN is not so distictly diffrent from N, or, sorting benefits the downstream pipeline, it is better to sort. Otherwise, if only focusing on who is the second, O(n) method is more efficent.

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
Finally, construct a random sample generater, which is used for 1000 trials(could be larger if you set another number) to test if the two methods reach the same result.

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
    high_bound = BinarySearch(age_list, high_age)
    low_bound = BinarySearch(age_list, low_age)
    if age_list[high_bound] >= high_age:
        high_bound +=1
    if age_list[low_bound] < low_age:
        low_bound -=1
    return low_bound-high_bound+1


def N_search_methd(age_list, high_age, low_age):
    high_bound, low_bound = 0, 0
    for i in range(len(age_list)):
        if age_list[i] < high_age:
            high_bound = i
            break
    for i in range(len(age_list)):
        if age_list[i] < low_age:
            low_bound = i-1
            break
    return low_bound - high_bound + 1
        
for trial in range(10**3):
    random_Generator = 84*np.random.random(2)
    high_age, low_age = np.max(random_Generator), np.min(random_Generator)
    assert age_scale_idf(age_list, high_age, low_age) == N_search_methd(age_list, high_age, low_age), 'test failed'

print('1000 tests success')

```
```
>>> 1000 tests success
```
As the above shows, the tester with the random sample generator goes through all 1000 trials, and those look all fine. You could set the trial number to a even larger one like 1e10000.

## Modify the above, including possibly the data structure you're using, to provide a function that returns both the total number of patients in an age range AND the number of males in the age range, all in O(log n) time as measured after any initial data setup. (2 points). Test it (show your tests) and justify that your algorithm works. (2 points)

Divid the DataFrame to two according to the gender, whether is male or not. Call age range identification with bisection function ```age_scale_idf``` twice and return the number in this range respectively. Then we get total number in this range and also the number of male in this range.


```python
def age_and_gender(data, low_age, high_age):
    data_notmale, data_male = data[data['gender']!='male'], data[data['gender'] == 'male']
    num_notmale, num_male = age_scale_idf(list(data_notmale['age']), high_age, low_age), age_scale_idf(list(data_male['age']), high_age, low_age)
    return f'Total number of age range between {low_age} and {high_age} is {num_notmale+num_male}, and {num_male} males are in this range. 
```
Test it 
```python
age_and_gender(sorted_age, 0.1, 41.5)
```
```
>>> 'Total number of age range between 41.5 and 100 is 150471, and 71308 males are in this range. '
