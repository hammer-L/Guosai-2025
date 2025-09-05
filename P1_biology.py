import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

df = pd.read_excel("filtered-ref-converted.xlsx") 
df_selected = pd.DataFrame({
    'Y染色体浓度': df.iloc[:, 21].tolist(),  # V列
    '孕妇BMI': df.iloc[:, 10].tolist(),        # J列
    '孕周数值': df.iloc[:, 9].tolist(), 
    '13号染色体的Z值': df.iloc[:, 16].tolist() # Q列
})
X = df_selected[['孕妇BMI', '孕周数值', '13号染色体的Z值']].values
y = df_selected['Y染色体浓度'].values
def logit_func(X, b0, b1, b2, b3):
    x1, x2, x3 = X
    return 1 / (1 + np.exp(-(b0 + b1*x1 + b2*x2 + b3*x3)))
y_scaled = (y - y.min()) / (y.max() - y.min())
X_train, X_test, y_train, y_test = train_test_split(
    X,y_scaled, test_size=0.3, random_state=42)

params, _ = curve_fit(
    logit_func,
    (X_train[:,0], X_train[:,1], X_train[:,2]),
    y_train,
    p0=[0.1, 0.1, 0.1, 0.1]  # 初始参数
)

# 预测
y_train_pred = logit_func((X_train[:,0], X_train[:,1], X_train[:,2]), *params)
y_test_pred = logit_func((X_test[:,0], X_test[:,1], X_test[:,2]), *params)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Logit模型参数:", params)
print("训练集 MSE:", train_mse)
print("测试集 MSE:", test_mse)