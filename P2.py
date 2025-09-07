import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import itertools

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 数据读取与预处理 ----------
df = pd.read_excel("ref.xlsx", sheet_name="男胎检测数据")

def parse_gestation(week_str):
    try:
        if isinstance(week_str, str) and "w" in week_str:
            parts = week_str.split("w")
            week = int(parts[0])
            day = int(parts[1].replace("+", "").strip()) if "+" in week_str else 0
            return week + day/7
        elif isinstance(week_str, (int, float)):
            return week_str
    except:
        return None
    return None

df["孕周数"] = df["检测孕周"].apply(parse_gestation)
df["BMI"] = df["孕妇BMI"]

# ---------- 仅保留 BMI >= 37.5 的样本 ----------
data = df[["孕周数", "BMI", "Y染色体浓度"]].dropna()
data = data[data["BMI"] >= 37.5].copy()

# ---------- 自适应分箱，保证每个 bin 至少有 2 个样本 ----------
max_bins = 5
num_bins = min(max_bins, data.shape[0] // 2)  # 每个bin至少2个样本
if num_bins < 2:  # 如果样本太少，就不分层
    stratify_col = None
else:
    data['FF_bin'] = pd.cut(data['Y染色体浓度'], bins=num_bins, labels=False)
    counts = data['FF_bin'].value_counts()
    valid_bins = counts[counts >= 2].index
    data = data[data['FF_bin'].isin(valid_bins)]
    stratify_col = data['FF_bin']

# ---------- 划分训练集和测试集 ----------
train_data, test_data = train_test_split(
    data, test_size=0.3, random_state=42, stratify=stratify_col
)

G_train, B_train, Y_train = train_data["孕周数"].values, train_data["BMI"].values, train_data["Y染色体浓度"].values
G_test,  B_test,  Y_test  = test_data["孕周数"].values,  test_data["BMI"].values,  test_data["Y染色体浓度"].values

# ---------- 候选特征定义 ----------
candidates = {
    'x1':    lambda G,B: G,
    'x2':    lambda G,B: B,
    '1/x1':  lambda G,B: np.where(G!=0, 1/G, np.nan),
    '1/x2':  lambda G,B: np.where(B!=0, 1/B, np.nan),
    'x1^2':  lambda G,B: G**2,
    'x2^2':  lambda G,B: B**2,
    'e^x1':  lambda G,B: np.exp(G),
    'e^x2':  lambda G,B: np.exp(B),
    '07^x1': lambda G,B: np.power(0.7, G),
    '07^x2': lambda G,B: np.power(0.7, B)
}

# ---------- 特征子集穷举搜索 ----------
feature_names = list(candidates.keys())
best_r2 = -np.inf
best_combo = None
best_model = None
best_X_test = None
best_Y_pred = None

for r in range(1, len(feature_names)+1):
    for combo in itertools.combinations(feature_names, r):
        X_train = []
        X_test = []
        valid = True
        for feat in combo:
            f_train = candidates[feat](G_train, B_train)
            f_test  = candidates[feat](G_test,  B_test)
            if np.any(np.isnan(f_train)) or np.any(np.isnan(f_test)):
                valid = False
                break
            X_train.append(f_train)
            X_test.append(f_test)
        if not valid:
            continue
        X_train = np.vstack(X_train).T
        X_test  = np.vstack(X_test).T

        model = LinearRegression().fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        r2 = r2_score(Y_test, Y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_combo = combo
            best_model = model
            best_X_test = X_test
            best_Y_pred = Y_pred

# ---------- 输出最优隐函数 ----------
coef = best_model.coef_
intercept = best_model.intercept_
equation = "Y = {:.4f}".format(intercept)
for c, f in zip(coef, best_combo):
    equation += " + ({:.4f} * {})".format(c, f)
print(f"最佳特征组合: {best_combo}, 测试集R² = {best_r2:.4f}")
print("拟合得到的隐函数：")
print(equation)

# ---------- 可视化：真实 vs 预测 ----------
plt.figure(figsize=(6,6))
plt.scatter(Y_test, best_Y_pred, alpha=0.7, edgecolor="k")
plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()],
         "r--", lw=2, label="理想拟合线")
plt.xlabel("真实 Y 值")
plt.ylabel("预测 Y 值")
plt.title(f"最佳组合 {best_combo}\nR² = {best_r2:.4f}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
