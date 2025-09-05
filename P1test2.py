import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------
# 读取数据
# ----------------
male_df = pd.read_excel("filtered-ref.xlsx", sheet_name="Sheet1")


# ----------------
# 数据预处理
# ----------------
def parse_gestation(week_str):
    try:
        if isinstance(week_str, str) and "w" in week_str:
            parts = week_str.split("w")
            week = int(parts[0])
            day = int(parts[1].replace("+", "").strip()) if "+" in week_str else 0
            return week + day / 7
        elif isinstance(week_str, (int, float)):
            return week_str
    except:
        return None
    return None


male_df["孕周数"] = male_df["检测孕周"].apply(parse_gestation)
male_df["BMI"] = male_df["体重"] / ((male_df["身高"] / 100) ** 2)
male_clean = male_df[["孕周数", "BMI", "Y染色体浓度"]].dropna()

X = male_clean[["孕周数", "BMI"]]
y = male_clean["Y染色体浓度"]

# ----------------
# 划分训练集/测试集
# ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------
# 定义回归模型
# ----------------
models = {
    "线性回归": LinearRegression(),
    "随机森林回归": RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42),
    "XGBoost回归": xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42),
}

# ----------------
# 训练并绘制散点图
# ----------------
plt.figure(figsize=(8, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    plt.scatter(y_test, y_pred, alpha=0.6, label=f"{name} (R²={r2:.2f})")

# 理想预测线 y=x
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label="理想预测")

plt.xlabel("真实 Y染色体浓度")
plt.ylabel("预测 Y染色体浓度")
plt.title("连续 Y染色体浓度回归预测 - 真实 vs 预测")
plt.legend()
plt.grid(True)
plt.show()

import statsmodels.api as sm

# ----------------
# 线性回归显著性检验
# ----------------
X_sm = sm.add_constant(X)  # 添加截距项
ols_model = sm.OLS(y, X_sm).fit()

print(ols_model.summary())
