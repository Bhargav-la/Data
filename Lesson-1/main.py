import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

titanic = pd.read_csv("Titanic Dataset.csv", sep="\t")

titanic.head()
print(titanic.head())
titanic.columns
print(titanic.columns)
print(titanic.shape)

print(titanic.isnull().sum())

print(plt.figure(figsize=(10,6)))

sns.heatmap(titanic.isnull(), cmap="spring")

print(plt.show())

print(titanic.drop("Cabin", axis=1, inplace=True))

print(titanic.head())

print(titanic.dropna(inplace=True))

print(titanic.isnull().sum())

plt.figure(figsize=(10, 6))

sns.heatmap(titanic.isnull(), cbar=False)

plt.show()

print(pd.get_dummies(titanic["Sex"]).head())

sex = pd.get_dummies(titanic["Sex"], drop_first=True)

print(sex.head(4))

embarked = pd.get_dummies(titanic["Embarked"], drop_first=True)

print(embarked.head(4))

pclass = pd.get_dummies(titanic["Pclass"], drop_first = True)

print(pclass.head(4))

titanic = pd.concat([titanic, sex, embarked, pclass],axis = 1)

print(titanic.head())