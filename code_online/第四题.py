import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn import tree
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体，优先使用SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 修复孕周转换函数
def convert_gestational_week(x):
    # 统一转换为小写处理
    x_lower = x.lower()
    if '+' in x_lower:
        # 分割周和天
        parts = x_lower.split('w')
        if len(parts) > 1:
            weeks = float(parts[0])
            days_part = parts[1].split('+')
            if len(days_part) > 1:
                days = float(days_part[1])
                return weeks + days/7
    # 如果没有'+'或者格式不正确，尝试直接提取周数
    try:
        return float(x_lower.split('w')[0])
    except:
        # 如果所有尝试都失败，返回NaN
        return float('nan')

female = pd.read_csv(r"C:\Users\86137\Desktop\国赛\女胎儿.csv")
female['labels'] = np.where(female['染色体的非整倍体'].isna(), 0, 1)
female.drop(['序号','孕妇代码','末次月经','检测日期','染色体的非整倍体','胎儿是否健康'],axis=1,inplace=True)
female['孕周数值'] = female['检测孕周'].apply(convert_gestational_week)
female.drop("检测孕周",axis=1,inplace=True)
print(female)

female.dropna(inplace=True)
print(female.info())

labels = female['labels']
female = female.drop("labels",axis=1)
print(female)

female_one = pd.get_dummies(female,dtype='int8')
X = female_one.values
y = labels.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

dt_classifier = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  # 控制树深度防止过拟合
    min_samples_split=5,
    min_samples_leaf=2
)
# 训练模型
dt_classifier.fit(X_resampled, y_resampled)
y_pred = dt_classifier.predict(X_test)
y_pred_proba = dt_classifier.predict_proba(X_test)  # 获取预测概率
#计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 计算ROC曲线和AUC值
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])  # 计算FPR, TPR和阈值
roc_auc = auc(fpr, tpr)
# 5. 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 绘制随机猜测的参考线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)  # 添加网格线

# 显示图形
plt.show()

unique_labels = np.unique(labels)
class_names = [f'类别{label}' for label in unique_labels]
# 可视化决策树
plt.figure(figsize=(20, 12))
plot_tree(
    dt_classifier,
    filled=True,  # 填充颜色表示类别
    feature_names=female_one.columns,
    class_names=class_names,
    rounded=True,  # 圆角矩形
    proportion=True,  # 显示比例而非样本数
    precision=2  # 数值精度
)
plt.title("决策树可视化 - 胎儿异常分类")
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

feature_importances = dt_classifier.feature_importances_
features = female_one.columns

# 创建特征重要性DataFrame并按重要性排序
importance_df = pd.DataFrame({
    'feature': features,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("\n特征重要性排序:")
print(importance_df)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('特征重要性')
plt.title('决策树特征重要性')
plt.gca().invert_yaxis()  # 重要性从高到低显示
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()