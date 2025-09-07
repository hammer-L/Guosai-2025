import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
data = df[["孕周数", "BMI", "Y染色体浓度"]].dropna()

# 分层抽样划分训练集和测试集
num_bins = 5
data = data.copy()
data['FF_bin'] = pd.cut(data['Y染色体浓度'], bins=num_bins, labels=False)
train_data, test_data = train_test_split(
    data, test_size=0.3, random_state=42, stratify=data['FF_bin']
)

G_train, B_train, Y_train = train_data["孕周数"].values, train_data["BMI"].values, train_data["Y染色体浓度"].values
G_test,  B_test,  Y_test  = test_data["孕周数"].values,  test_data["BMI"].values,  test_data["Y染色体浓度"].values

# ---------- 候选特征定义 ----------
candidates = {
    'x1':    lambda G,B: G,
    'x2':    lambda G,B: B,
    '1/x1':lambda G,B: np.where(G!=0, 1/G, np.nan),
    '1/x2':lambda G,B: np.where(B!=0, 1/B, np.nan),
    'x1^2': lambda G,B: G**2,
    'x2^2': lambda G,B: B**2,
    'e^x1':lambda G,B: np.exp(G),
    'e^x2':lambda G,B: np.exp(B),
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
    for combo in __import__('itertools').combinations(feature_names, r):
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
import matplotlib.pyplot as plt

# 创建一个独立的 Figure 和 Axes 对象，尺寸与之前相同
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制散点图
ax.scatter(Y_test, best_Y_pred, alpha=0.7)

# 绘制理想拟合线（y=x）
ax.plot([Y_test.min(), Y_test.max()],
        [Y_test.min(), Y_test.max()],
        "r--", lw=2)

plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
plt.gca().set_facecolor('#F8F9FA')

    # 美化边框
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('#DDDDDD')

# 设置轴标签和标题
ax.set_xlabel("真实 Y染色体浓度")
ax.set_ylabel("预测 Y染色体浓度")
ax.set_title(f"隐函数模型\nR²={best_r2:.3f}") # 假设 R² 值为 r2_impl

# 设置网格线
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
plt.gca().set_facecolor('#F8F9FA')

plt.show()


