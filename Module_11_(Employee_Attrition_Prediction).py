import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from collections import Counter
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

train = pd.read_csv("C:/Users/15/Desktop/데이터셋/[Dataset]_Module11_Train_(Employee).csv")
print(train.columns)

print(train.shape)
print(train.head())

print(train.info())

print(train.dtypes)

j=0
for i in train.columns:
    print(i,":",np.unique(train[train.columns[j]].values).size)
    j = j+1

# print(train.nunique())# train 데이터 세트 유니크 아이템의 개수를 확인합니다.

plt.hist(train['Attrition_rate'])
plt.show() # Attrition_rate 컬럼에 대하 히스토그램 그리기

#성별이 직원의 성과에 미치는 영향

df_value = train.groupby('Gender')['growth_rate'].mean()
df_index = train.groupby('Gender')['Gender']
df = pd.DataFrame(train.groupby('Gender')['growth_rate'].mean())
print(df)

colors = ['y', 'dodgerblue']
plt.bar(np.unique(train['Gender']),df_value, color = colors, width=0.4)
plt.title('Growth_Rate')
plt.show()

plt.bar(np.unique(train['Gender']),train.groupby('Gender')['Gender'].count(), color = colors, width=0.4)
plt.title('Comparison of Males and Females')
print(train.groupby('Gender')['Gender'].count())
plt.show()

print(train.groupby('Hometown')['Hometown'].count())
Hometown_groups = train.groupby('Hometown')['Hometown'].count()
Hometown_groups.plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of various groups')
plt.ylabel('count')
plt.xlabel('Group')
plt.xticks(rotation=0)
plt.show() #막대그래프 그리기

print(train.groupby('Relationship_Status')['Relationship_Status'].count())
Hometown_groups = train.groupby('Relationship_Status')['Relationship_Status'].count()
Hometown_groups.plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of various groups')
plt.ylabel('count')
plt.xlabel('Group')
plt.xticks(rotation=0)
plt.show() #막대그래프 그리기

df_age_value = train.groupby('Relationship_Status')['Attrition_rate'].mean()
df_age_index = train.groupby('Relationship_Status')['Relationship_Status']

df_age = pd.DataFrame(train.groupby('Relationship_Status')['Attrition_rate'].mean())
print(df_age)


df_age.T.plot(kind='bar', figsize=(10, 6))#.T를 넣으니까 라벨위치 ?? 바뀜 궁금하면 지워보기
plt.title('Relationship_Status')
plt.ylabel('Rates')
plt.xticks(rotation=0)
plt.show() #막대그래프 그리기

print(train.describe())

print(train.isna().any())

plt.figure(figsize=(18,10))

cor = train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Accent)
plt.show()
plt.savefig("main_correlation.png")

label = ["Attrition_rate"]
features = ['VAR7','VAR6','VAR5','VAR1','VAR3','growth_rate','Time_of_service','Time_since_promotion','Travel_Rate','Post_Level','Education_Level']

featured_data = train.loc[:,features+label]
print(featured_data.shape)

featured_data_drop_na = featured_data.dropna(axis = 0)
print(featured_data_drop_na.shape)

X = featured_data_drop_na.loc[:,features]
y = featured_data_drop_na.loc[:,label]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.45)
print('x_train values count:', X_train.shape)
print('y_train values count:', y_train.shape)
print('x_test values count:', X_test.shape)
print('y_test values count:', y_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

y_test_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)

# score 를 출력해 봅니다. : error(MAE, RMSLE)
def show_scores(y_test, val_preds):
    scores = {"Valid MAE": mean_absolute_error(y_test, val_preds),
              "Valid RMSLE": rmsle(y_test, val_preds)}
    return scores
def rmsle(y_test, y_preds):
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

print(show_scores(y_test, y_test_predict))

test = pd.read_csv("C:/Users/15/Desktop/데이터셋/[Dataset]_Module11_sample_(Employee).csv")
ID          = ["Employee_ID"]
pred_data   = test.loc[:,features+ID]
pred_data   = pred_data.dropna(axis=0)
y = pred_data.loc[:,ID]
sample_data = test.loc[:,features]
sample_data = sample_data.dropna(axis=0)

y_hat = model.predict(sample_data)

size = len(y_hat)
c=[]
for i in range(len(y_hat)):
    c.append((y_hat[i][0].round(5)))
pf=c[:size]

# 예측
dff = pd.DataFrame({'Employee_ID':y['Employee_ID'],'Attrition_rate':pf})
dff.head()

dff.head(20).sort_values(by='Attrition_rate', ascending=False)