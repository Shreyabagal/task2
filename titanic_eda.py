# titanic_eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create Image folder if it doesn't exist
os.makedirs("Image", exist_ok=True)

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Data Cleaning
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Convert categorical columns
train['Sex'] = train['Sex'].astype('category')
train['Embarked'] = train['Embarked'].astype('category')

# Drop irrelevant columns
train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 1. Age & Fare Distribution
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(train['Age'], kde=True, bins=30)
plt.title('Age Distribution')

plt.subplot(1, 2, 2)
sns.histplot(train['Fare'], kde=True, bins=30)
plt.title('Fare Distribution')

plt.tight_layout()
plt.savefig("Image/age_fare_distribution.png")
plt.close()

# 2. Categorical Distributions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Sex', data=train)
plt.title('Gender Distribution')

plt.subplot(1, 2, 2)
sns.countplot(x='Embarked', data=train)
plt.title('Embarked Distribution')

plt.tight_layout()
plt.savefig("Image/sex_embarked_distribution.png")
plt.close()

# 3. Survival Rate Analysis
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Sex', y='Survived', data=train)
plt.title('Survival Rate by Gender')

plt.subplot(1, 2, 2)
sns.barplot(x='Pclass', y='Survived', data=train)
plt.title('Survival Rate by Class')

plt.tight_layout()
plt.savefig("Image/survival_rate_analysis.png")
plt.close()

# 4. Correlation Heatmap
numeric_data = train.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("Image/correlation_matrix.png")
plt.close()

# 5. Pairplot
sns.pairplot(train[['Age', 'Fare', 'Pclass', 'Survived']])
plt.savefig("Image/pairplot.png")
plt.close()
