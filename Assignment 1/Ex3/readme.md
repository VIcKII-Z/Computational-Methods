# Ex3 @Weijin ZOU


All the code blocks and results can be tested and viewed in [```HW1-Ex3.ipynb```](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%201/Ex3/HW1-Ex3.ipynb)

## Data 
* data source:  
@The New York Times    {The New York Times. (2021). Coronavirus (Covid-19) Data in the United States. Retrieved https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv, from https://github.com/nytimes/covid-19-data.}
* download time:   
2021-9-10  

## Make a function that takes a list of state names and plots their cases vs date using overlaid line graphs, one for each selected state.
Firstly, I created a pivot table out of the source dataframe to extract useful information. Also, I did some simple data cleaning.
```python
merged_covid = pd.pivot_table(covid_table,index=['state'],columns=['date'],values=['cases']).fillna(0).T
```
Then, below is the body of the function. __I personally use the increase of new cases instead of just total cases in the source data.__
```python
def cases_date_by_state(merged_covid, state_list,**n):
    for state in state_list:
        state_info = merged_covid[state].unstack().T
        new_cases = np.diff(state_info['cases'])
        plt.plot(state_info.index[1:], new_cases)
    plt.legend([state for state in state_list])
    plt.xticks(state_info.index[[1,100, 200, 300,400,500,len(state_info.index)-1]], rotation=90)
    plt.xlabel('date')
    plt.ylabel('new cases')
    plt.show()
```
## Test the above function and provide examples of it in use
Specifically, I choose Alabama and Alaska to test my function and below are the test code and its output figure.
```
# test code 
state_list = ['Alabama','Alaska']
cases_date_by_state(merged_covid, state_list)
```
![](https://github.com/VIcKII-Z/BIS634/blob/main/Assignment%201/Ex3/Unknown)

## Make a function that takes the name of a state and returns the date of its highest number of new cases.

Below are the body of main function and its test code. 
```python
def max_new_cases(state):
    state_info = merged_covid[state].unstack().T
    new_cases = np.diff(state_info['cases'])
    index = np.argmax(new_cases)   
    return state_info.T.columns[index+1]

# test code
max_new_cases('Connecticut')
```
```
>>> '2020-12-28'
```
## Make a function that takes the names of two states and reports which one had its highest number of cases first and how many days separate that one's peak from the other one's peak
To illustrate, I use the average number of new cases in 7 days to measure the number of cases mentioned in the assignment, because the data from us-states.csv is cumulative case number which does not match the statement in the assignment(_Connecticut had its highest number of cases in mid-January_) and thus cannot be used directly. This measurement is also consistant with what was used by New York Times.   

It is also notewothy that the output of my funtion is the exact statement of the answer to this question, which illustrates the two states, the separate days of its peaks and its choronological order. 
```python
from dateutil.parser import parse as parse_date


def average_increase_7_peak(state):
    state_info = merged_covid[state].unstack().T
    new_cases = np.diff(state_info['cases'])
    ave_7 = np.zeros(7)
    for i in range(7, len(state_info.index)):
        ave_7 = np.append(ave_7, np.mean(new_cases[i-6:i+1]))
    index = np.argmax(ave_7)
    
    return state_info.T.columns[index+1]


def report_2_states(state1, state2):
    date1 = parse_date(average_increase_7_peak(state1))
    date2 = parse_date(average_increase_7_peak(state2))
    sep = (date2 - date1).days
    print(f'{state1 if sep>=0 else state2} had its highest number of cases {sep} days before {state2 if sep>0 else state1}.')
```
## Test the above function and provide examples of it in use
To test the above function, I use Colorado and Connecticut. And you could see the result is quite readable and matches the statement in assignment.  
In addtion, you could use __average_increase_7_peak__ alone to get the exact date of a state's highest average number of new cases in 7 days.
```python
#test code
report_2_states('Colorado','Connecticut')
```
```
>>> Colorado had its highest number of cases 55 days before Connecticut.
```
```python
average_increase_7_peak('Connecticut')
```
```
>>> '2021-01-13'
```


