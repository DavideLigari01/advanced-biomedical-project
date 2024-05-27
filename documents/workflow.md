# What we have done until now

## 1. EDA

We have done some exploratory data analysis on the dataset to identify the class distribution, outliers and data visualization within classes

## 2. Feature extraction

We have extracted some features from the dataset, we have extracted the following features:

- MFCC
  -Chroma
  -RMS
  -ZCR
  -CQT
  -Spectral Centroid
  -Spectral Bandwidth
  -Spectral Roll-off

### Sample rate and interval selection

We have selected a sample rate of 4000 Hz and an interval of 2 seconds for the feature extraction

## 3. Feature selection

### 3.1 Optimal number of features for each type of feature

We have selected the following number of features for each type of feature:

| Type                | N° Features |
| ------------------- | ----------- |
| MFCC                | 30          |
| CQT                 | 70          |
| Chroma              | 12          |
| RMS                 | 40          |
| Zero Crossing Rates | 40          |
| Spectral Centroid   | 40          |
| Spectral Bandwidth  | 60          |
| Spectral Rolloff    | 40          |

### 3.2 Balancing method selection

We have selected the following balancing methods:

- No balancing
- Prior balancing
- Posterior balancing
- Both balancing -> OPTIMAL

### 3.3 Correlation analysis

#### 3.3.1 Correlation thresholds selection

f1_score macro: (0.0, 0.6, 30) - 0.8646068456432318

#### 3.3.2 Correlation analysis and feature selection

We have selected the following features:

MFCC: 25
Chroma: 12
CQT: 1
RMS: 0
ZCR: 1
SC: 0
SB: 0
SR: 0

Features kept : ['MFCC 2', 'MFCC 3', 'MFCC 11', 'MFCC 12', 'MFCC 13', 'MFCC 14',
'MFCC 15', 'MFCC 16', 'MFCC 17', 'MFCC 18', 'MFCC 19', 'MFCC 20',
'MFCC 21', 'MFCC 22', 'MFCC 23', 'MFCC 24', 'MFCC 25', 'MFCC 26',
'MFCC 27', 'MFCC 28', 'MFCC 29', 'MFCC 30', 'Chroma 2', 'CQT 7',
'ZCR 5', 'label'],

## 4. Model selection

We have selected the following models:

- Random Forest
- MLP -> OPTIMAL
- CatBoost
- XGBoost

## 5. Hyperparameter tuning

We have performed hyperparameter tuning on the selected model

## 6. Model evaluation and analysis

