# Ex3 @Weijin ZOU

## Data 
data source:@nytimes https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv.  
download time: 2021-9-10  

## Make a function that takes a list of state names and plots their cases vs date using overlaid line graphs, one for each selected state.
Below is the body of the function. __Personally, I use increase of new cases.__
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
[]()






