import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
male_df = pd.read_excel("ref.xlsx", sheet_name="男胎检测数据")

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

coef = pd.DataFrame({
    "变量": ["孕周数","BMI"],
    "系数": np.round(log_reg.coef_[0], 4)
})
intercept = round(log_reg.intercept_[0], 4)

print("模型系数：")
print(coef)
print("截距：", intercept)
# ----------------
# 随机森林参数
# ----------------

rf_clf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42)
rf_clf.fit(X, y)
y_pred_prob_rf = rf_clf.predict_proba(X)[:,1]
auc_rf = roc_auc_score(y, y_pred_prob_rf)

# ----------------
# 绘制 随机森林 ROC 曲线
# ----------------
fpr_rf, tpr_rf, _ = roc_curve(y, y_pred_prob_rf)
plt.figure(figsize=(6,6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest ROC (AUC = {auc_rf:.3f})')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.show()

# ----------------
# 绘制 线性预测 ROC 曲线
# ----------------
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"Logistic回归 (AUC = {auc_score:.3f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("假阳性率 (FPR)")
plt.ylabel("真正率 (TPR)")
plt.title("ROC曲线 - Y染色体浓度是否达标预测")
plt.legend(loc='lower right')  # 只调用一次，指定位置
plt.grid(True)
plt.show()



