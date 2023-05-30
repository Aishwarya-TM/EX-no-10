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
~df.duplicated()
df1=df[~df.duplicated()]
df1
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
df1.drop([0,1,2],axis=1, inplace=True)
df1
sns.barplot(x="Species",y="SepalLengthCm",data=df1)
plt.xticks(rotation = 90)
plt.show()
sns.lineplot(x="PetalLengthCm",y="PetalWidthCm",data=df1,hue="Species",style="Species")
sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm",hue="Species",data=df1)
sns.histplot(data=df1, x="SepalLengthCm", hue="Species", element="step", stat="density")
sns.relplot(data=df1,x=df1["PetalWidthCm"],y=df1["PetalLengthCm"],hue="Species")

