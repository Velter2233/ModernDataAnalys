import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score

import statsmodels.api as sm

# 1. Подготовка данных

print("1. Загрузка данных...")
df = pd.read_excel("Online Retail.xlsx")

# Удаление пропусков и некорректных значений
df.dropna(subset=['CustomerID'], inplace=True)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Создание целевой переменной
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print("Данные очищены")
print("-" * 50)

# 2. Разведочный анализ данных

print("2. Общая информация:")
print(df.info())
print("-" * 50)

print("3. Количественные оценки:")
stats = df[['Quantity', 'UnitPrice', 'TotalPrice']].describe().T
stats['variance'] = df[['Quantity', 'UnitPrice', 'TotalPrice']].var()
print(stats)
print("-" * 50)

print("4. Корреляционный анализ:")
corr = df[['Quantity', 'UnitPrice', 'TotalPrice']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Матрица корреляции")
plt.show()

print(corr['TotalPrice'].sort_values(ascending=False))
print("-" * 50)

print("5. Поиск выбросов (Boxplot):")
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[['Quantity', 'UnitPrice', 'TotalPrice']])
plt.title("Выбросы в данных")
plt.xticks(rotation=45)
plt.show()

# Расчитывание IQR для TotalPrice
Q1 = df['TotalPrice'].quantile(0.25)
Q3 = df['TotalPrice'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['TotalPrice'] < lower) | (df['TotalPrice'] > upper)]
print(f"Количество выбросов TotalPrice: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
print("-" * 50)

# 3. Линейная регрессия

print("6. Линейная регрессия")

target = 'TotalPrice'

# Модель 1 Quantity + UnitPrice 
features_full = ['Quantity', 'UnitPrice']

X = df[features_full]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

model_full = sm.OLS(y_train, X_train_sm).fit()
print("\nМОДЕЛЬ 1 (Quantity + UnitPrice)")
print(model_full.summary())

y_pred_full = model_full.predict(X_test_sm)
rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_full))
r2_full = r2_score(y_test, y_pred_full)

# Модель 2 только UnitPrice 
features_reduced = ['UnitPrice']

X_train_red = X_train[features_reduced]
X_test_red = X_test[features_reduced]

X_train_red_sm = sm.add_constant(X_train_red)
X_test_red_sm = sm.add_constant(X_test_red)

model_red = sm.OLS(y_train, X_train_red_sm).fit()
print("\nМОДЕЛЬ 2 (только UnitPrice)")
print(model_red.summary())

y_pred_red = model_red.predict(X_test_red_sm)
rmse_red = np.sqrt(mean_squared_error(y_test, y_pred_red))
r2_red = r2_score(y_test, y_pred_red)

# Сравнение моделей
comparison = pd.DataFrame({
    'Metric': ['R2', 'RMSE'],
    'Model 1 (Full)': [r2_full, rmse_full],
    'Model 2 (Reduced)': [r2_red, rmse_red]
})
print("\nСравнение моделей:")
print(comparison)

# Визуализация предсказаний
subset = 1000
plt.figure(figsize=(14, 6))
plt.plot(y_test.iloc[:subset].values, label='Реальные', linewidth=2)
plt.plot(y_pred_full.iloc[:subset].values, label='Model 1', linestyle='--')
plt.plot(y_pred_red.iloc[:subset].values, label='Model 2', linestyle=':')
plt.legend()
plt.title("Реальные vs Предсказанные значения")
plt.show()

# Остатки для модели 2
plt.figure(figsize=(10, 5))
sns.histplot(y_test - y_pred_red, kde=True)
plt.title("Распределение остатков (Модель 2)")
plt.xlabel("Ошибка")
plt.show()
