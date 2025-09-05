import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lifelines import KaplanMeierFitter, CoxPHFitter
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体，优先使用SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

male = pd.read_csv(r"C:\Users\86137\Desktop\国赛\男胎儿.csv")
male.drop('染色体的非整倍体',axis=1,inplace=True)
male.dropna(inplace=True)

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

male['孕周数值'] = male['检测孕周'].apply(convert_gestational_week)
male.drop("检测孕周",axis=1,inplace=True)
print(male.head())

labels = male['Y染色体浓度'] >= 0.04
male_label = male[labels].copy()
print(male_label.head())

first = (male_label
             .sort_values(['孕妇代码', '孕周数值'])
             .drop_duplicates(subset=['孕妇代码'], keep='first'))
first.drop(['序号','检测日期','胎儿是否健康'],axis=1,inplace=True)
print(first)

reach_table = first.copy()
X =reach_table.drop(["孕妇代码","末次月经"],axis=1)
X = pd.get_dummies(X,dtype='int8')
X = X.values
print(X)

inertias = []
k_range = range(1, 15)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)  # inertia_是簇内平方和属性
# 绘制碎石图
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-', markersize=8)
plt.xlabel('K值（簇的数量）', fontsize=12)
plt.ylabel('簇内平方和（Inertia）', fontsize=12)
plt.title('手肘法确定最佳K值', fontsize=14)
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

km = KMeans(n_clusters=5, random_state=42, n_init='auto')
reach_table['BMI组'] = km.fit_predict(X)

cuts = pd.qcut(reach_table['孕妇BMI'], q=5, precision=1)
reach_table['BMI区间'] = cuts

group_stats = (reach_table
               .groupby('BMI区间')['孕周数值']
               .agg(['count', 'median', lambda x: x.quantile(0.9)]))
group_stats.rename(columns={'<lambda_0>':'P90孕周'}, inplace=True)
print(group_stats)

idx = group_stats.index.tolist()   # 先拷贝成可变列表
idx[0]  = "<29.4"
idx[-1] = "≥33.9"
group_stats.index = idx            # 重新赋回去
print(group_stats)

bmi_intervals = group_stats.index
p90_weeks = group_stats['P90孕周']
counts = group_stats['count']
# 创建图形
plt.figure(figsize=(10, 6))
# 绘制气泡图
# 气泡大小需要缩放，这里将count乘以一个系数使其在图中显示合适
bubble_sizes = [count * 30 for count in counts]  # 调整系数使气泡大小合适
# 为每个BMI区间创建气泡
scatter = plt.scatter(range(len(bmi_intervals)), p90_weeks, s=bubble_sizes, alpha=0.6, c='steelblue')
# 设置图表标题和标签
plt.title('BMI区间与P90孕周关系气泡图', fontsize=16)
plt.xlabel('BMI区间', fontsize=12)
plt.ylabel('P90孕周', fontsize=12)
# 设置横坐标刻度
plt.xticks(range(len(bmi_intervals)), bmi_intervals)
# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)
# 添加数值标签
for i, (week, count) in enumerate(zip(p90_weeks, counts)):
    plt.annotate(f'P90: {week:.1f}\nCount: {count}',
                 (i, week),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center',
                 fontsize=10)
# 添加图例说明气泡大小代表样本数量
# 创建一些示例气泡用于图例
for count in [15, 20, 25]:
    plt.scatter([], [], s=count*50, alpha=0.6, c='steelblue', label=f'Count: {count}')
plt.legend(scatterpoints=1, frameon=True, labelspacing=1.5, title='样本数量', loc='best')
plt.tight_layout()
plt.show()