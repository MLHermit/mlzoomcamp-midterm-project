
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


data = pd.read_parquet(r"C:\Users\Abdullahi Mujaheed\Downloads\0000.parquet")


data.isna().sum()


data


data.describe()


data.columns, data.dtypes


data.timestamp = pd.to_datetime(data.timestamp)


data.describe()

 [markdown]
# the data spans from october 12th, 2024 to october 12th, 2025. Exactly one year


data.fatalities.value_counts(), data.severity.value_counts(), data.cause.value_counts()


data.head()


data['month'] = data.timestamp.dt.month


monthly_counts =data.month.value_counts().sort_index()


monthly_counts.values


plt.bar(monthly_counts.index, monthly_counts.values)
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents by Month')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation= 45)
plt.grid()
plt.show()

 [markdown]
# the least month of road accidents is February with roughly 10700, while the month with the most road accidents is August with over 12000 accidents.


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.tree import DecisionTreeClassifier


vect = DictVectorizer(sparse=False)


data_cat = data[['severity', 'cause']]


data_cat


cat_vect = vect.fit_transform(data_cat.to_dict(orient='records'))


client = [{'timestamp': '2025-09-01 15:16:08', 'lat': 7.681686, 'lon': 12.088482, 'severity': 'severe', 'cause': 'mechanical', 'vehicles_involved': 3, 'injuries': 4}]


def predict(client):
    cl = pd.DataFrame(client)
    cl['month'] = pd.to_datetime(cl['date']).dt.month
    client_dicts = cl[['severity', 'cause']].to_dict(orient='records')
    client_cat = vect.transform(client_dicts)
    vect_data = pd.DataFrame(client_cat, columns=vect.get_feature_names_out())
    r = cl.drop(['severity', 'cause'], axis=1)
    f = r.join(vect_data)
    pred=log_reg.predict(f)
    return pred


predict(client)











cat_vect


df_cat = pd.DataFrame(cat_vect, columns=vect.get_feature_names_out())


data_dropped = data.drop(columns=['severity', 'cause', 'incident_id'],)


data_dropped


sns.heatmap(data_dropped.corr(), annot=True, cmap= 'coolwarm')

 [markdown]
# No correlation between the features whatsoever


full_data =data_dropped.join(df_cat)


full_data


data['fatalities'].groupby(data['month']).mean().hist(bins=12)


data[data['month']==2]['fatalities'].value_counts()


y = full_data['fatalities'].values
X = full_data.drop(columns=['fatalities', 'timestamp']).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .25, random_state=42)


log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)


log_reg.fit(X_train, y_train)


y_log_pred = log_reg.predict(X_test)
y_log_pred


np.unique(y_log_pred), np.unique(y_test)


rmse(y_test, y_log_pred)





dtc = DecisionTreeClassifier(max_depth=2, random_state=42)


dtc.fit(X_train, y_train)


from sklearn.utils.validation import check_is_fitted


y_pred_dtc = dtc.predict(X_test)
print(np.unique(y_pred_dtc, return_counts=True))


print(dtc.feature_importances_)


rmse(y_test, y_pred_dtc)


rmses = {}
for depth in tqdm(np.arange(2, 22, 2)):
    dtcs = DecisionTreeClassifier(max_depth= depth, random_state=2)
    dtcs.fit(X_train, y_train)
    y_dtcs = dtcs.predict(X_test)
    rmse_dtc = rmse(y_test, y_dtcs)
    rmses[depth] = rmse_dtc


rmses

 [markdown]
# since the best rmse score with the decision tree is the same with that of the logistic regression, we save the logistic regression


import pickle as pkl


with open(r'C:\Users\Abdullahi Mujaheed\Desktop\mlzoom\mlzoomcamp\first_pipenv\vectorizer_and_model', 'wb') as f_out:
    pkl.dump((vect, log_reg), f_out)





