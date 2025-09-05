import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------
# 数据读取与预处理
# ----------------
male_df = pd.read_excel("ref.xlsx", sheet_name="男胎检测数据")


def parse_gestation(week_str):
    try:
        if isinstance(week_str, str) and "w" in week_str:
            parts = week_str.split("w")
            week = int(parts[0])
            day = int(parts[1].replace("+", "").strip()) if "+" in parts[1] else 0
            return week + day / 7
        elif isinstance(week_str, (int, float)):
            return week_str
    except:
        return None
    return None


male_df["孕周数"] = male_df["检测孕周"].apply(parse_gestation)
male_df["BMI"] = male_df["体重"] / ((male_df["身高"] / 100) ** 2)
male_clean = male_df[["孕周数", "BMI", "Y染色体浓度"]].dropna()  # 保留所有样本

# ----------------
# BMI分组
# ----------------
bins = [20, 28, 32, 36, 40, np.inf]
labels = ['20-28', '28-32', '32-36', '36-40', '40+']
male_clean['BMI组'] = pd.cut(male_clean['BMI'], bins=bins, labels=labels, right=False)

# ----------------
# 定义回归模型
# ----------------
models = {
    '线性回归': LinearRegression(),
    '随机森林回归': RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42),
    'XGBoost回归': xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
}

# ----------------
# 按BMI组绘制真实值 vs 预测值散点图
# ----------------
for bmi_group in labels:
    idx = male_clean['BMI组'] == bmi_group
    df_group = male_clean.loc[idx]
    X_group = df_group[['孕周数', 'BMI']]
    y_group = df_group['Y染色体浓度']

    if len(y_group) < 5:  # 样本太少跳过
        continue

    # 划分组内训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_group, y_group, test_size=0.3, random_state=42)

    plt.figure(figsize=(6, 6))

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        plt.scatter(y_test, y_pred, label=f'{name} (R^2={r2:.2f})', alpha=0.7)

    # 绘制理想预测线 y=x
    plt.plot([y_group.min(), y_group.max()],
             [y_group.min(), y_group.max()], 'k--', label='理想预测')

    plt.xlabel("真实 Y染色体浓度")
    plt.ylabel("预测 Y染色体浓度")
    plt.title(f"BMI组 {bmi_group} - 回归预测对比")
    plt.legend()
    plt.grid(True)
    plt.show()
