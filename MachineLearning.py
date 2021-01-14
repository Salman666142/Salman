import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

data = pd.read_excel('MalwareSamples2.xlsx', index_col=0)
# data2 = pd.read_csv("MalwareSamples10000.csv", sep=",")
# print(data.head())
# data = data[
# ["hasZip", "hasPDF", "hasDoc", "hasUnknown", "hasURL", "urlCount",
# "senderDomainSuffix", "headerCount", "isMalware"]]
df = pd.DataFrame(data, columns=["hasZip", "hasPDF", "hasDoc", "hasUnknown", "hasURL", "isMalware"])

df['isMalware'] = (df['isMalware'] == "Yes") * 1
data["isMalware"] = df["isMalware"]

df["hasZip"] = (df["hasZip"] == "Yes") * 1
data["hasZip"] = df["hasZip"]

df["hasPDF"] = (df["hasPDF"] == "Yes") * 1
data["hasPDF"] = df["hasPDF"]

df["hasDoc"] = (df["hasDoc"] == "Yes") * 1
data["hasDoc"] = df["hasDoc"]

df["hasUnknown"] = (df["hasUnknown"] == "Yes") * 1
data["hasUnknown"] = df["hasUnknown"]

df["hasURL"] = (df["hasURL"] == "Yes") * 1
data["hasURL"] = df["hasURL"]

domain_suffix = {'com.au': 1, 'edu.au': 2, 'co.uk': 3, 'com': 4, 'net': 5, '.in': 6, 'other': 7, 'net.au': 8}
data["senderDomainSuffix"] = [domain_suffix[item] for item in data["senderDomainSuffix"]]

predict = "isMalware"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

logistic = linear_model.LogisticRegression()
logistic.fit(x_train, y_train)
acc2 = logistic.score(x_test, y_test)

RandomForest = RandomForestClassifier()
RandomForest.fit(x_train, y_train)
acc3 = RandomForest.score(x_test, y_test)

print("Linear Regression accuracy =", acc * 100, "%")
print("Logistic Regression accuracy =", acc2 * 100, "%")
print("Random Forest accuracy =", acc3 * 100, "%")
