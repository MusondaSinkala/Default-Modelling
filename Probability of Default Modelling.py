#%% Intro

"""
Created on Sat Aug 20 20:27:51 2022

@author: MusondaSinkala

Re: Probablity of Default Modelling

"""

#%% Package Importation and Console setup

# Standard data manipulation and vizualization packages
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Package for outlier detection
from scipy import stats

# Model-building and validation packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                   OneHotEncoder, LabelEncoder)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sns.set_style("whitegrid")

os.chdir(r"C:\\Users\\MusondaSinkala\\Documents\\Personal")

#%% Data Loading

# Read in data and print the first 5 rows in the dataset

data = pd.read_excel("Deloitte Assessment DRI - Data.xlsx")

data.head()

#%% Data Cleaning - removing nulls and reindexing

# Remove any unnecessary columns from the dataset
data = data.drop("Unnamed: 0", axis = 1)

data.head()

# Detect any columns with null values

data.info()

# Drop any rows with null values

data = data.dropna()

data = data.reset_index()

data = data.drop('index', axis = 1)

data.info()

#%% Histogram and Boxplot vizualization

def visualize(feat):
    plt.figure(figsize = (16, 4))
    
    plt.suptitle(feat)
    
    plt.subplot(1, 2, 1)
    plt.title("Histogram")
    data[feat].hist(bins = 50)

    plt.subplot(1, 2, 2)
    plt.title("Box Plot")
    sns.boxplot(x = data["DefaultInd"], y = data[feat])

#%% Further Data Cleaning - cleaning categorical features and outlier removal

# Customer distribution by default status
data['DefaultInd'].value_counts()

data.nunique()

# Customer distribution by Gender
data['Gender'].value_counts()

# Unique values for Payment Method variable
data['PaymentMethod'].unique()

# Unique values for Spending Target variable
data['SpendingTarget'].unique()
# Two study loan spending methods were identified - thus they need to be combined into one
data['SpendingTarget'].value_counts()
data['SpendingTarget'] =  data['SpendingTarget'].replace(to_replace = 'Study loan debt',
                                                         value = 'Study')
data['SpendingTarget'].value_counts()

# Distribution by Nationality
data['Nationality'].value_counts()
# Duplicate nationalities identified - needs resolving
data['Nationality'] = data['Nationality'].str.lower()
data['Nationality'].value_counts()

# Outlier detection

# Identify the numeric features in the dataset
num_feat = list(data.select_dtypes(include = ['float64', 'int64']))

# Vizualize distribution of numeric features
for i in range(len(num_feat)):
    visualize(num_feat[i])

# Calculate z scores for all numeric features
zscores = np.abs(data[num_feat].apply(stats.zscore))
# detect outliers
outliers = list(zscores[zscores > 3].dropna(thresh = 1).index)
# drop outliers
data.drop(data.index[outliers], inplace = True)

# Detect further outliers
for i in range(len(num_feat)):
    visualize(num_feat[i])

data = data.reset_index()
data = data.drop('index', axis = 1)

cat_feat = data.drop(['DefaultInd', 'CustomerID'],
                     axis = 1).select_dtypes(include = ['object']).columns

model_data = data.reset_index()
model_data = model_data.drop('index', axis = 1)
model_data[cat_feat] = model_data[cat_feat].apply(lambda x: x.factorize()[0])


data['DefaultInd'].value_counts()
data['Gender'].value_counts()
data['Nationality'].value_counts()

#%% Exploratory Data Analysis - Correlation

# Correlation Detection
correlation = round(model_data[num_feat].corr(), 2)
sns.set(rc = {'figure.figsize':(15, 8)})
sns.heatmap(correlation,
            xticklabels = correlation.columns,
            yticklabels = correlation.columns,
            annot = True)
plt.savefig("correlation.png", dpi = 2500)

#%% Exploratory Data Analysis and Data Cleaning - Categorical Variables




# for feat in cat_feat:
#     pd.Series(data[feat].factorize())

data['MaritalStatus'].value_counts()

# Relationship between default status and employment status
((pd.pivot_table(data,
                 index = 'EmploymentStatus',
                 values = 'CustomerID',
                 columns = 'DefaultInd',
                 aggfunc = 'count'))/np.array(data['DefaultInd'].value_counts())) * 100

((pd.pivot_table(data,
                 index = 'BKR_Registration',
                 values = 'CustomerID',
                 columns = 'DefaultInd',
                 aggfunc = 'count'))/np.array(data['DefaultInd'].value_counts())) * 100

pd.pivot_table(data,
                 index = 'BKR_Registration',
                 values = 'CustomerID',
                 columns = 'DefaultInd',
                 aggfunc = 'count')

pd.pivot_table(data,
                 index = 'EmploymentStatus',
                 values = 'CreditAmount',
                 aggfunc = 'mean')

#%% Exploratory Data Analysis and Data Cleaning - Numeric Variables


# sns.distplot(data['EstimatedIncome'], hist = True, color = '#86BC25')
# data['EstimatedIncome'].plot.hist(grid = True, bins = 20, rwidth = 0.9,
#                                   color = '#86BC25')

# sns.distplot(data['CreditAmount'], hist = True, color = '#86BC25')
# data['CreditAmount'].plot.hist(grid = True, bins = 75, rwidth = 0.9,
#                                color = '#86BC25')

# plt.scatter(data['CurrentLoans'], data['MonthlyCharges'])

### Histogram of Income by Default Status
sns.distplot(data['EstimatedIncome'][data['DefaultInd'] == 0], hist = False,
             color = '#86BC25', label = 'Not Defaulted')
sns.distplot(data['EstimatedIncome'][data['DefaultInd'] == 1], hist = False,
             color = '#000000', label = 'Defaulted')
# plt.ylim(0, 1.8)
plt.title('Histogram of Estimated Income by Default Status')
plt.xlabel = 'Estimated Income in Euros'
plt.legend(loc = 'upper right')
plt.savefig("correlation.png", dpi = 2500)


### Histogram of Income by Payment Method
sns.distplot(data['EstimatedIncome'][data['PaymentMethod'] == 'Bank transfer'],
             hist = False,
             color = '#86BC25', label = 'Bank transfer')
sns.distplot(data['EstimatedIncome'][data['PaymentMethod'] == 'Mailed check'],
             hist = False,
             color = '#000000', label = 'Mailed check')
sns.distplot(data['EstimatedIncome'][data['PaymentMethod'] == 'Electronic check'],
             hist = False,
             color = '#0076A8', label = 'Electronic check')
sns.distplot(data['EstimatedIncome'][data['PaymentMethod'] == 'Credit card'],
             hist = False,
             color = '#97999B', label = 'Credit card')
# plt.ylim(0, 1.8)
plt.title('Histogram of Estimated Income by Payment Method')
plt.xlabel = 'Estimated Income in Euros'
plt.legend(loc = 'upper right')


### Histogram of CreditAmount by Default Status
sns.distplot(data['CreditAmount'][data['DefaultInd'] == 0], hist = False,
             color = '#86BC25', label = 'Not Defaulted')
sns.distplot(data['CreditAmount'][data['DefaultInd'] == 1], hist = False,
             color = '#000000', label = 'Defaulted')
# plt.ylim(0, 1.8)
plt.title('Histogram of Credit Amount by Default Status')
plt.xlabel = 'Credit Amount in Euros'
plt.legend(loc = 'upper right')

data.describe()

#%% Splitting Dataset

x = model_data.drop(['DefaultInd', 'CustomerID'], axis = 1)
y = model_data['DefaultInd']
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.3,
                                                    random_state = 1)

#%% Model Building

dtree = DecisionTreeClassifier(max_depth = 4)

dtree.fit(x_train, y_train) 

predictions = dtree.predict(x_test)
print(confusion_matrix(y_true = y_test, y_pred = predictions))
print(classification_report(y_true = y_test, y_pred = predictions))

features = list(model_data.columns[1:])

# Vizualize decision tree

fig = plt.figure(figsize = (15, 10))
_ = tree.plot_tree(dtree, 
                   feature_names = features,  
                   class_names = ['Default', 'Not Default'],
                   filled = True,
                   rounded = True,
                   impurity = False,
                   proportion = True,
                   label = 'root',
                   fontsize = 15)
plt.savefig("dtree.png", dpi = 2500)


importances = pd.DataFrame({'FeatureNames': x_train.columns,
                            'Importance': dtree.feature_importances_})
importances = importances.sort_values(by = 'Importance', ascending = False)

plt.figure().set_facecolor('white')
plt.bar(importances['FeatureNames'][1:10],
        importances['Importance'][1:10],
        color = '#86BC25')

plt.savefig("importance.png", dpi = 2500)
