import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

male_df = pd.read_excel("filref.xlsx", sheet_name="男胎检测数据")

# ----------------
# 数据预处理
# ----------------
# 处理孕周字符串，如 "11w+6" -> 11 + 6/7
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

male_df["孕周数"] = male_df["检测孕周"].apply(parse_gestation)
male_df["BMI"] = male_df["体重"] / ((male_df["身高"]/100) ** 2)

# 保留关键字段
male_clean = male_df[["孕周数","BMI","Y染色体浓度"]].dropna()

# ----------------
# 构造分类变量：是否达标 (Y浓度 >= 4%)
# ----------------
male_clean["达标"] = (male_clean["Y染色体浓度"] >= 0.04).astype(int)

# 特征与标签
X = male_clean[["孕周数", "BMI"]]
y = male_clean["达标"]

# ----------------
# 逻辑回归建模
# ----------------
log_reg = LogisticRegression()
log_reg.fit(X, y)

# 预测概率
y_pred_prob = log_reg.predict_proba(X)[:, 1]

# ROC 曲线和 AUC
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
auc_score = roc_auc_score(y, y_pred_prob)

# ----------------
# 绘制 ROC 曲线
# ----------------
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"Logistic回归 (AUC = {auc_score:.3f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("假阳性率 (FPR)")
plt.ylabel("真正率 (TPR)")
plt.title("ROC曲线 - Y染色体浓度是否达标预测")
plt.legend()
plt.grid(True)
plt.show()

# ----------------
# 输出模型系数
# ----------------

coef = pd.DataFrame({
    "变量": ["孕周数","BMI"],
    "系数": np.round(log_reg.coef_[0], 4)
})
intercept = round(log_reg.intercept_[0], 4)

print("模型系数：")
print(coef)
print("截距：", intercept)

# ----------------
# 散点图 + 拟合曲线
# ----------------
plt.figure(figsize=(10,6))

# 散点图：Y浓度 vs 孕周，颜色表示BMI
scatter = plt.scatter(male_clean['孕周数'], male_clean['Y染色体浓度'],
                      c=male_clean['BMI'], cmap='viridis', alpha=0.6)
plt.axhline(y=0.04, color='red', linestyle='--', linewidth=1.5, label='Y浓度 = 4%')
plt.colorbar(scatter, label='BMI')
plt.xlabel('孕周数')
plt.ylabel('Y染色体浓度')
plt.title('Y染色体浓度 vs 孕周数 (按BMI显示)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))

# 散点图：Y浓度 vs BMI，颜色表示孕周
scatter = plt.scatter(male_clean['BMI'], male_clean['Y染色体浓度'],
                      c=male_clean['孕周数'], cmap='plasma', alpha=0.6)
plt.axhline(y=0.04, color='red', linestyle='--', linewidth=1.5, label='Y浓度 = 4%')
plt.colorbar(scatter, label='孕周数')
plt.xlabel('BMI')
plt.ylabel('Y染色体浓度')
plt.title('Y染色体浓度 vs BMI (按孕周显示)')
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# 三维散点
sc = ax.scatter(male_clean['BMI'], male_clean['孕周数'], male_clean['Y染色体浓度'],
                c=male_clean['Y染色体浓度'], cmap='viridis', s=20, alpha=0.6)

# 添加颜色条
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('Y染色体浓度')

# 添加坐标轴标签
ax.set_xlabel('BMI')
ax.set_ylabel('孕周数')
ax.set_zlabel('Y染色体浓度')
ax.set_title('BMI、孕周数与Y染色体浓度三维散点图')

plt.show()
