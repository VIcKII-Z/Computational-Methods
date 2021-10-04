# EX3
## Identify a data set online (10 points) that you find interesting that could potentially be used for the final project; 
I use data from Kaggle competition: https://www.kaggle.com/c/ventilator-pressure-prediction.

## Describe the dataset (10 points) 
There are 2 files. One for training and the other one for testing.
```
Files
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
Below is the rules of data access and use clarified by the host.
>Data Access and Use. You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.
