# Ex:10 Data Science Process on Complex Dataset
# AIM:
To Perform Data Science Process on a complex dataset and save the data to a file

# ALGORITHM:
STEP 1 Read the given Data 

STEP 2 Clean the Data Set using Data Cleaning Process 

STEP 3 Apply Feature Generation/Feature Selection Techniques on the data set 

STEP 4 Apply EDA /Data visualization techniques to all the features of the data set

# CODE:
## Data Cleaning Process:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as plt

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("Iris.csv")

df.head(10)

df.info()

df.describe()

df.isnull().sum()

## Handling Outliers:

q1 = df['SepalLengthCm'].quantile(0.25)

q3 = df['SepalLengthCm'].quantile(0.75)

IQR = q3 - q1

print("First quantile:", q1, " Third quantile:", q3, " IQR:", IQR, "\n")

lower = q1 - 1.5 * IQR

upper = q3 + 1.5 * IQR

outliers = df[(df['SepalLengthCm'] >= lower) & (df['SepalLengthCm'] <= upper)]

from scipy.stats import zscore

z = outliers[(zscore(outliers['SepalLengthCm']) < 3)]

print("Cleaned Data: \n")

print(z)

## EDA Techniques:

df.skew()

df.kurtosis()

sns.boxplot(x="SepalLengthCm",data=df)

sns.boxplot(x="SepalWidthCm",data=df)

sns.countplot(x="Species",data=df)

sns.distplot(df["PetalWidthCm"])

sns.distplot(df["PetalLengthCm"])

sns.histplot(df["SepalLengthCm"])

sns.histplot(df["PetalWidthCm"])

sns.scatterplot(x=df['SepalLengthCm'],y=df['SepalWidthCm'])

import matplotlib.pyplot as plt

states=df.loc[:,["Species","SepalLengthCm"]]

states=states.groupby(by=["Species"]).sum().sort_values(by="SepalLengthCm")

plt.figure(figsize=(17,7))

sns.barplot(x=states.index,y="SepalLengthCm",data=states)

plt.xlabel=("Species")

plt.ylabel=("SepalLengthCm")

plt.show()

import matplotlib.pyplot as plt

states=df.loc[:,["Species","PetalWidthCm"]]

states=states.groupby(by=["Species"]).sum().sort_values(by="PetalWidthCm")

plt.figure(figsize=(17,7))

sns.barplot(x=states.index,y="PetalWidthCm",data=states)

plt.xlabel=("Species")

plt.ylabel=("PetalWidthCm")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)

## Feature Generation:
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

le=LabelEncoder()

df['Id']=le.fit_transform(df['SepalLengthCm'])

df

S=['Iris-setosa','Iris-virginica','Iris-versicolor']

enc=OrdinalEncoder(categories=[S])

enc.fit_transform(df[['Species']])

df['SP1']=enc.fit_transform(df[['Species']])

df

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from google.colab import files

uploaded = files.upload()

from sklearn.preprocessing import OneHotEncoder

df1 = pd.read_csv("Iris.csv")

ohe=OneHotEncoder(sparse=False)

enc=pd.DataFrame(ohe.fit_transform(df1[['Species']]))

df1=pd.concat([df1,enc],axis=1)

df1

## Feature Transformation:
import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

sm.qqplot(df1['SepalLengthCm'],fit=True,line='45')

plt.show()

sm.qqplot(df1['SepalWidthCm'],fit=True,line='45')

plt.show()

import numpy as np

from sklearn.preprocessing import PowerTransformer

transformer=PowerTransformer("yeo-johnson")

df1['PetalLengthCm']=pd.DataFrame(transformer.fit_transform(df1[['PetalLengthCm']]))

sm.qqplot(df1['PetalLengthCm'],line='45')

plt.show()

transformer=PowerTransformer("yeo-johnson")

df1['PetalWidthCm']=pd.DataFrame(transformer.fit_transform(df1[['PetalWidthCm']]))

sm.qqplot(df1['PetalWidthCm'],line='45')

plt.show()

qt=QuantileTransformer(output_distribution='normal')

df1['SepalWidthCm']=pd.DataFrame(qt.fit_transform(df1[['SepalWidthCm']]))

sm.qqplot(df1['SepalWidthCm'],line='45')

plt.show()

## Data Visua/lization:
sns.barplot(x="Species",y="SepalLengthCm",data=df1)

plt.xticks(rotation = 90)

plt.show()

sns.lineplot(x="PetalLengthCm",y="PetalWidthCm",data=df1,hue="Species",style="Species")

sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm",hue="Species",data=df1)

sns.histplot(data=df1, x="SepalLengthCm", hue="Species", element="step", stat="density")

sns.relplot(data=df1,x=df1["PetalWidthCm"],y=df1["PetalLengthCm"],hue="Species")

# OUTPUT:
## Data Cleaning Process:
![1](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/cd6d0aa4-cddb-4753-8c7b-311d5165e099)

![2](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/ba794e7d-fb73-453a-8e64-5fd00c836fae)

![3](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/8590c027-c7e5-48e1-8f52-e4547c6092e6)

## Handling Outliers:
![4](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/d8c5b88f-c389-454c-ba8b-afaab812b423)

## EDA Techniques:
![6](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/3640af92-76e7-4fa5-8181-a1f0e04c90d5)

![7](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/a7ede59f-d379-41cf-900a-b1355db80b99)

![8](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/9fd77d49-a1f9-4b96-8f2a-a33f72933730)

![9](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/69a83512-26d9-43c1-8446-0602dda40b15)

![10](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/0a46a646-e0af-4060-bc09-6d0afa237b38)

![11](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/0fdbf330-19d5-4e7d-a5fc-609217d9aa40)

![12](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/14a9a518-d90b-4dbe-801f-d1499d8d4b61)

![13](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/10ae74a3-84a6-47fa-9eba-cb793b1fbf4c)

## Feature Generation:
![14](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/d97deca8-a37e-4a66-a82b-42b2cdd2c325)

![15](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/d86833b8-2bff-4332-88e2-f27701d79d58)

## Feature Transformation:
![16](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/b6a24e35-3f02-4810-a471-4c1a5e197679)

![17](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/5fc551ed-a8af-489d-9eac-592a6d7716b4)

## Data Visua/lization:
![18](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/54429932-27fe-40d6-baa6-cc651424ff74)

![19](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/5ea08c89-e43e-44c1-b2ef-cfd1a25875f1)

![20](https://github.com/Aishwarya-TM/EX-no-10/assets/127846109/de3e6763-925c-4ef7-8483-281829a7341b)

## RESULT:
Thus the Data Science Process on Complex Dataset were performed and output was verified successfully.


