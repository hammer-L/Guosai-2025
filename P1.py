import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

sns.set(style="whitegrid")
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
data = df[["孕周数","BMI","Y染色体浓度"]].dropna()

# ---------- 数据集分层划分（按Y浓度区间比例） ----------
num_bins = 5
data = data.copy()
data['FF_bin'] = pd.cut(data['Y染色体浓度'], bins=num_bins, labels=False)

train_data, test_data = train_test_split(
    data,
    test_size=0.3,
    random_state=42,
    stratify=data['FF_bin']
)

G_train, B_train, FF_train = train_data["孕周数"].values, train_data["BMI"].values, train_data["Y染色体浓度"].values
G_test, B_test, FF_test = test_data["孕周数"].values, test_data["BMI"].values, test_data["Y染色体浓度"].values
X_train_df, y_train_df = train_data[["孕周数", "BMI"]], train_data["Y染色体浓度"]
X_test_df, y_test_df = test_data[["孕周数", "BMI"]], test_data["Y染色体浓度"]

# ===============================================================
# 1. 非线性模型（仅测试集）
# ===============================================================
def FF_model(X, a, p, b, c):
    G, B = X
    A = a * G**p
    M = b * np.exp(c*B)
    return A / (A + M)
popt, pcov = curve_fit(FF_model, (G_train,B_train), FF_train, bounds=(0, [np.inf,np.inf,np.inf,np.inf]))
a,p,b,c = popt
FF_pred_nl = FF_model((G_test, B_test), a, p, b, c)
r2_nl = r2_score(FF_test, FF_pred_nl)
mse_nl = mean_squared_error(FF_test, FF_pred_nl)

# ===============================================================
# 2. C-logit 模型（仅测试集）
# ===============================================================
eps = 1e-6
FF_clip_train = np.clip(FF_train, eps, 1-eps)
logit_FF_train = np.log(FF_clip_train / (1-FF_clip_train))
X_logit_train = pd.DataFrame({
    "lnG": np.log(G_train),
    "BMI": B_train
})
X_logit_train = sm.add_constant(X_logit_train)
model_logit = sm.OLS(logit_FF_train, X_logit_train).fit()

G_test_logit = G_test
B_test_logit = B_test
FF_test_logit = np.clip(FF_test, eps, 1-eps)
X_test_logit = pd.DataFrame({
    "lnG": np.log(G_test_logit),
    "BMI": B_test_logit
})
X_test_logit = sm.add_constant(X_test_logit)
logit_pred_test = model_logit.predict(X_test_logit)
FF_pred_logit = 1 / (1 + np.exp(-logit_pred_test))
r2_logit = r2_score(FF_test_logit, FF_pred_logit)

# ===============================================================
# 3. 线性回归模型（仅测试集）
# ===============================================================
X_const_train = sm.add_constant(X_train_df)
model_lin = sm.OLS(y_train_df, X_const_train).fit()
X_test_const = sm.add_constant(X_test_df)
y_pred_lin = model_lin.predict(X_test_const)
r2_lin = r2_score(y_test_df, y_pred_lin)

# ===============================================================
# 4. 隐函数模型（仅测试集，可自定义项和自动权重数量）
# ===============================================================
# ---------- 通用隐函数特征构造器 ----------
def implicit_feature_builder(x1, x2, y, terms):
    feats = []
    for term in terms:
        if term == 'x1^2':
            feats.append(x1**2)
        elif term == 'x2^2':
            feats.append(x2**2)
        elif term == 'x1':
            feats.append(x1)
        elif term == 'x2':
            feats.append(x2)
        elif term == 'y':
            feats.append(y)
        elif term == '1':
            feats.append(np.ones_like(x1))
        elif term == '0.7^x2':
            feats.append(0.7 ** x2)
        elif term == '0.7^x1':
            feats.append(0.7 ** x1)
        elif term == '1/x1':
            feats.append(1 / x1)
        elif term == '1/x2':
            feats.append(1 / x2)

        # 可继续扩展更多项，如 log(x1)、sin(x1) 等
        else:
            raise ValueError(f"未知项: {term}")
    return np.column_stack(feats)
# 可自由扩展项，权重数量自动适配
#'x1', 'x1^2', 'x2^2', '07^x1', '07^x2'
implicit_terms = [
    'x1^2',
    'x2^2',
    'x1',
    'y',
    '1',
    '0.7^x1',
    '0.7^x2',

]

X_implicit = implicit_feature_builder(
    X_train_df['孕周数'].values,
    X_train_df['BMI'].values,
    y_train_df.values,
    implicit_terms
)
y_rhs = -y_train_df.values**2  # 右端项可自定义

implicit_model = LinearRegression(fit_intercept=False)
implicit_model.fit(X_implicit, y_rhs)

# 测试集预测
x1_test = X_test_df['孕周数'].values
x2_test = X_test_df['BMI'].values
y_test = y_test_df.values

X_test_implicit = implicit_feature_builder(x1_test, x2_test, y_test, implicit_terms)
coef_values = implicit_model.coef_

# 用字典自动映射每个项的权重
term_dict = dict(zip(implicit_terms, coef_values))

# 按隐函数关系解y（二次方程结构，可自由调整）
a = np.ones_like(x1_test)
b = term_dict.get('x1*y',0)*x1_test + term_dict.get('x2*y',0)*x2_test + term_dict.get('y',0)
c = term_dict.get('x1^2',0)*x1_test**2 + term_dict.get('x2^2',0)*x2_test**2 \
    + term_dict.get('x1*x2',0)*x1_test*x2_test + term_dict.get('x1',0)*x1_test \
    + term_dict.get('x2',0)*x2_test + term_dict.get('1',0) \
    + term_dict.get('exp(x1)',0)*np.exp(x1_test) \
    + term_dict.get('exp(x2)',0)*np.exp(x2_test) \
    + term_dict.get('exp(y)',0)*np.exp(y_test)

discriminant = b**2 - 4*a*c
discriminant[discriminant<0] = 0
y_pred1 = (-b + np.sqrt(discriminant)) / (2*a)
y_pred2 = (-b - np.sqrt(discriminant)) / (2*a)
y_mean = y_train_df.mean()
y_pred_impl = np.where(np.abs(y_pred1 - y_mean) < np.abs(y_pred2 - y_mean), y_pred1, y_pred2)
r2_impl = r2_score(y_test_df, y_pred_impl)

# ===============================================================
# 四图合并显示（全部只用测试集）
# ===============================================================
def create_and_save_plot(ax, x_data, y_data, title, file_name):
    """
    绘制并保存单个预测 vs. 真实值的散点图。
    """
    ax.scatter(x_data, y_data, alpha=0.7)
    # 绘制y=x对角线
    ax.plot([x_data.min(), x_data.max()], [x_data.min(), x_data.max()], 'r--', lw=2)

    ax.set_xlabel("真实 Y染色体浓度")
    ax.set_ylabel("预测 Y染色体浓度")
    ax.set_title(title)
    ax.grid(True)

    # 设置网格和样式
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.gca().set_facecolor('#F8F9FA')

    # 美化边框
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#DDDDDD')

    # 调整布局并保存图片
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close(ax.figure)  # 关闭图形以释放内存


# 1. 线性回归
fig_lin, ax_lin = plt.subplots(figsize=(8, 8))
title_lin = f"线性回归\nR²={r2_lin:.3f}"
file_name_lin = "linear_regression_plot.png"
create_and_save_plot(ax_lin, y_test_df, y_pred_lin, title_lin, file_name_lin)

# 2. C-logit
fig_logit, ax_logit = plt.subplots(figsize=(8, 8))
title_logit = f"FF稀释比例模型C-logit形式\nR²={r2_logit:.3f}"
file_name_logit = "c_logit_plot.png"
create_and_save_plot(ax_logit, FF_test_logit, FF_pred_logit, title_logit, file_name_logit)

# 3. 非线性
fig_nl, ax_nl = plt.subplots(figsize=(8, 8))
title_nl = f"FF稀释比例模型\nR²={r2_nl:.3f}"
file_name_nl = "nonlinear_model_plot.png"
create_and_save_plot(ax_nl, FF_test, FF_pred_nl, title_nl, file_name_nl)

# 4. 隐函数
fig_impl, ax_impl = plt.subplots(figsize=(8, 8))
title_impl = f"隐函数模型\nR²={r2_impl:.3f}"
file_name_impl = "implicit_function_plot.png"
create_and_save_plot(ax_impl, y_test_df, y_pred_impl, title_impl, file_name_impl)

print("四张独立的模型预测 vs. 真实值图片已成功保存。")