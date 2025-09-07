import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import norm
import itertools
import seaborn as sns

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

# ---------- 计算 Y 染色体浓度标准误 ----------
def compute_y_error(df, conc_col='Y染色体浓度', n_col='唯一比对的读段数  ', alpha=0.2):
    z = norm.ppf(1 - alpha/2)
    p = df[conc_col]
    N = df[n_col]

    se = np.sqrt(p*(1-p)/N)
    se[N <= 0] = np.nan
    se[p.isna()] = np.nan

    df['SE_Y'] = se
    df['CI_lower'] = (p - z*se).clip(0,1)
    df['CI_upper'] = (p + z*se).clip(0,1)
    return df

df = compute_y_error(df)
df["孕周数"] = df["检测孕周"].apply(parse_gestation)

# ---------- 准备特征 ----------
features = ['孕周数', '身高', '体重', '年龄', 'SE_Y']  # 加入孕周
data = df[features + ['Y染色体浓度']].dropna().copy()  # Y染色体浓度作为目标变量

# 分层抽样
num_bins = 5
data['FF_bin'] = pd.cut(data['Y染色体浓度'], bins=num_bins, labels=False)
train_data, test_data = train_test_split(
    data, test_size=0.3, random_state=42, stratify=data['FF_bin']
)

G_train = train_data['孕周数'].values
H_train = train_data['身高'].values
W_train = train_data['体重'].values
A_train = train_data['年龄'].values
SE_train = train_data['SE_Y'].values
Y_train = train_data['Y染色体浓度'].values

G_test = test_data['孕周数'].values
H_test = test_data['身高'].values
W_test = test_data['体重'].values
A_test = test_data['年龄'].values
SE_test = test_data['SE_Y'].values
Y_test = test_data['Y染色体浓度'].values

# ---------- 候选特征定义 ----------
candidates = {
    '孕周数':   lambda G,H,W,A,SE: G,
    '身高':     lambda G,H,W,A,SE: H,
    '体重':     lambda G,H,W,A,SE: W,
    '年龄':     lambda G,H,W,A,SE: A,
    # 'SE_Y':   lambda G,H,W,A,SE: SE,  # 可选择保留
    '1/孕周数': lambda G,H,W,A,SE: np.where(G!=0, 1/G, np.nan),
    '1/身高':   lambda G,H,W,A,SE: np.where(H!=0, 1/H, np.nan),
    '1/体重':   lambda G,H,W,A,SE: np.where(W!=0, 1/W, np.nan),
    '孕周数^2': lambda G,H,W,A,SE: G**2,
    '身高^2':   lambda G,H,W,A,SE: H**2,
    '体重^2':   lambda G,H,W,A,SE: W**2,
    'e^孕周数': lambda G,H,W,A,SE: np.exp(G),
    'e^身高':   lambda G,H,W,A,SE: np.exp(H),
    'e^体重':   lambda G,H,W,A,SE: np.exp(W)
}

feature_names = list(candidates.keys())

# ---------- 特征子集穷举搜索 ----------
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
            f_train = candidates[feat](G_train, H_train, W_train, A_train, SE_train)
            f_test  = candidates[feat](G_test,  H_test,  W_test,  A_test,  SE_test)
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
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(8, 8))

# 绘制散点图和趋势线
sns.regplot(
    x=Y_test,
    y=best_Y_pred,
    scatter_kws={"color": "#fc5185", "alpha": 0.6},  # 散点参数
    line_kws={"color": "#A23B72", "lw": 2, 'label':'trend'},         # 拟合线参数
    ci=None,  # 不显示置信区间
    ax=ax,
)

# 绘制理想拟合线（y=x）
ax.plot(
    [Y_test.min(), Y_test.max()],
    [Y_test.min(), Y_test.max()],
    "--",color='#2E86AB', lw=2, label="Ideal Line"
)

# 设置标签和标题
ax.set_xlabel("True Y concentration", fontsize=12)
ax.set_ylabel("Predicted Y concentration", fontsize=12)
ax.set_title(f"Best Combination\nR²={best_r2:.3f}", fontsize=14, fontweight="bold")

# 设置网格和样式
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
plt.gca().set_facecolor('#F8F9FA')

# 美化边框
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('#DDDDDD')
# 优化外观
ax.legend()

plt.show()