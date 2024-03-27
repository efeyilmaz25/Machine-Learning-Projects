import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("hitters.csv")
df = df.dropna()
print(df)

print(df)
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.describe().T)


categorical_columns = df.select_dtypes(include=['object']).columns

le = LabelEncoder()
df[categorical_columns] = df[categorical_columns].apply(lambda col: le.fit_transform(col.astype(str)))

y = df["Salary"]
X = df.drop(["Salary"], axis=1)

ozellikler = X.columns
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

print("Mean Salary: ", df["Salary"].mean())
print("Std: ", df["Salary"].std())



def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

columns = [column for column in df.columns if "Salary" not in column]

for col in columns:
    plot_numerical_col(df, col)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
linear_model = LinearRegression().fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = linear_model.score(X_test, y_test)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)



plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Real vs Prediction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Coreect Prediction')
plt.xlabel('Real Values')
plt.ylabel('Estimated Values')
plt.title('Real vs Estimated Values')
plt.legend()
plt.show()

print("=============================")
#RMSE
print("RMSE: ", np.mean(np.sqrt(-cross_val_score(linear_model, X, y, cv=10, scoring="neg_mean_squared_error"))))


correlation_matrix = df.corr()

plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, annot_kws={'size': 10}),
plt.title('Attribute Correlation Map')
plt.show()

