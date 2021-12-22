# Ventilator Pressure Prediction
## Motivation
When a patient has problems breathing, what do doctors do? A ventilator is used to pump oxygen into the lungs of a sedated patient through a tube in the windpipe. Mechanical ventilation, on the other hand, is a clinician-intensive operation, which was highlighted during the early days of the COVID-19 epidemic. At the same time, even before clinical trials, finding novel methods for operating mechanical ventilators is prohibitively expensive. High-quality simulations have the potential to lower this barrier.

Simulators are currently trained as an ensemble, with each model simulating a single lung setting. However, because lungs and their features occupy a continuous space, a parametric technique that takes into account the variances in patient lungs must be investigated.
## Outline
This roport consist of 5 parts to illustrate on the pipline of predicting the pressure out of the ventilator, which are Dataset, Data Analysis, Analysis API,  Feature Engineering, Model Building and Result and Discussion.
## Dataset
The dataset is credit to Kaggle, which can be downloaded at https://www.kaggle.com/c/ventilator-pressure-prediction/data. It has three files, training, testing, and submission sample. The data analysis below are based on training set.

The training set has 6 variables, 5 of which are predictors and 1 is the target. The details are as belows.  
```R``` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change R by changing the diameter of the straw, with higher R being harder to blow.   
```C``` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change C by changing the thickness of the balloon’s latex, with higher C having thinner latex and easier to blow.   
```time_step```- the actual time stamp.   
```u_in``` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.   
```u_out``` - the control input for the exploratory solenoid valve. Either 0 or 1.   
```pressure(target)``` - the airway pressure measured in the respiratory circuit, measured in cmH2O. Breath_id is globally-unique time step for breaths.  
## Data Analysis
### Validation checking
Check missing data or mismatching data:
```python
df = pd.read_csv("ventilator-pressure-prediction/train.csv", na_values = ['--', 'N/A', 'na'])
print(df.isnull().sum())
```
```
id           0
breath_id    0
R            0
C            0
time_step    0
u_in         0
u_out        0
pressure     0
dtype: int64
```
![image](https://user-images.githubusercontent.com/62388643/147003118-6a62cc10-5d85-486f-b0a0-e5f2b9254603.png)

### Data overview and summary
Obviously this is a time series dataset, and each sample has 80 time steps.
```python
train.groupby('breath_id')['id'].count()
```
```
breath_id
1         80
2         80
3         80
4         80
5         80
          ..
125740    80
125742    80
125743    80
125745    80
125749    80
Name: id, Length: 75450, dtype: int64
```

U_out is the control input for the exploratory solenoid valve. It is either 0 or 1.
```python
train['u_out'].value_counts()
```
```
1    3745032
0    2290968
Name: u_out, dtype: int64
```
The remains statistical summary is as follows.
```
train[['R','C','u_in','pressure']].describe()
```
```

             R	           C	           u_in	         pressure
count	6.036000e+06	6.036000e+06	6.036000e+06	6.036000e+06
mean	2.703618e+01	2.608072e+01	7.321615e+00	1.122041e+01
std	1.959549e+01	1.715231e+01	1.343470e+01	8.109703e+00
min	5.000000e+00	1.000000e+01	0.000000e+00	-1.895744e+00
25%	5.000000e+00	1.000000e+01	3.936623e-01	6.329607e+00
50%	2.000000e+01	2.000000e+01	4.386146e+00	7.032628e+00
75%	5.000000e+01	5.000000e+01	4.983895e+00	1.364103e+01
max	5.000000e+01	5.000000e+01	1.000000e+02	6.482099e+01
```

R and C both have only three values, 20, 50, 5 and 50, 20, 10 respectively.
```python
train_ori['R'].unique(), train_ori['C'].unique() 
```
```
>>> (array([20, 50,  5]), array([50, 20, 10]))
```
The target pressure suggests a long-tail distribution.
```python

plt.title('Histogram of Train Pressures',size=14)
plt.hist(train_gf.sample(100_000).pressure.to_array(),bins=100)
plt.show()
print('Max pressure =',train_gf.pressure.max(), 'Min pressure =',train_gf.pressure.min())
```
![image](https://user-images.githubusercontent.com/62388643/147003347-834766b5-5653-42af-ab27-0436656d088b.png)
### Correlation

For each timestep, the most correlated variables are U_in(0.76) and U_out(-0.81).  
  
![image](https://user-images.githubusercontent.com/62388643/147006103-4d945e41-3c60-4bbe-b364-21e91a483fea.png)

## Analysis API
Also, I built a interface for the users and researchers to anlyze and visualize the related data.
The home page's look is as below. The data source could be reached by clicking the picture in the middle, and users could choose to analyze one of the variables in the dropdown.
![image](https://user-images.githubusercontent.com/62388643/147033850-70b1fb3d-dfef-4a7d-a706-a3b86285f9a1.png)
After choosing the varibles, the page will goes like this, showing the plot and its basic statistics. And users could return to homepage by clicking the left buttom text. Also, if other actions taken except for a valid varible input, the pages will turn to be a 404 notice page.
![image](https://user-images.githubusercontent.com/62388643/147034223-5f58b2cd-8b2d-448d-8f8a-c887effe1843.png)


## Feature Engineering
After looking into the original dataset, I focused on feature engineering. Features can be divided into original features and engineered features.   
For the original features, all features except R and C were used as they are, and one-hot encoded R and C were used to show the type information of R and C to the model. Encoding R and C to the one hot dummies are as follows.
```python

#Translate the R-C relation to categorial variables

df['R'] = df['R'].astype(str)
df['C'] = df['C'].astype(str)
df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
df = pd.get_dummies(df)
```
And the samples look like this.
![image](https://user-images.githubusercontent.com/62388643/147005685-60c3edd6-2763-42d5-8eec-914c06e709c6.png)




For engineered features, I did a lot of EDA and tried to put features the model could not see. In this context, I have created the following features:
```u_in_lagX```:  u_in value at next X time step
```u_in_lag_backX```:  u_in value at previous X time step
```u_in_diffX```: gap between the current u_in and that of next X time step 
```u_out_diffX```: gap between the current u_out and that of next X time step 


Details are below.
```python
df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
df = df.fillna(0)
df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']

df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']    

df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']

df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
```
## Model 
### Model Design
Based on its inherent characteristics of time series data, I used Bidrectional LSTM, which I usually used in Natural Language Process settings. Bi-LSTM has its algorithm-based strength to memorize the information of the both the next and previous neros instead of just only the current ones.

K-Folds is simply used as the validation strategy. 4 layers of Bi_LSTM are used for there are lag data with a window of four.
Besides, early stopping strategy is used and the final parameters are saved in ```foldX.hdf5```.
```python
EPOCH = 300
BATCH_SIZE = 1024
NUM_FOLDS = 10

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
test_preds = []
for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train[train_idx], train[test_idx] #8:2
        y_train, y_valid = targets[train_idx], targets[test_idx]
        
        model = keras.models.Sequential([
            keras.layers.Input(shape=train.shape[-2:]),
            keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.Dense(128, activation='selu'),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mae")
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
        es = EarlyStopping(monitor="val_loss", patience=60, verbose=1, mode="min", restore_best_weights=True)
         checkpoint_filepath = f"folds{fold}.hdf5"
        sv = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch',
            options=None
        )

        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr, es, sv])
        test_preds.append(model.predict(test).squeeze().reshape(-1, 1).squeeze())
```
## Result and disccussion
![image](https://user-images.githubusercontent.com/62388643/147009068-3a35ef3b-65c8-4f4e-baf6-b5842064da63.png)
Model early Stopped at 171/300 epoch. Validation loss is .16934, Test loss is .1576, and the top 10's test loss is .102

When reflecting on this project, I found there are some points which could be improved. For feature engineering, there are still some future work to do like data augmentation(Masknig augmentation/Shuffling augmentation), which could make the training data learns more about the real mechanism and performs better on the test data.




    

        




