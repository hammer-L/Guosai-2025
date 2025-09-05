
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from matplotlib.gridspec import GridSpec
from copulas.multivariate import GaussianMultivariate

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体，优先使用SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_theme(style="white", font_scale=1.2)

df = pd.read_excel(r'E:\0workspace\Guosai-2025\filtered-ref-converted.xlsx')

bins = [27, 31.25, 33.5, 38]
labels = ['low', 'mid', 'high']

df['bmi_group'] = pd.cut(df['孕妇BMI'], bins=bins, labels=labels, right=True)

# 查看每个区间的数量
print(df['bmi_group'].value_counts())

# 获取各分组的数据
df_low = df[df['bmi_group'] == 'low'].copy()
df_mid = df[df['bmi_group'] == 'mid'].copy()
df_high = df[df['bmi_group'] == 'high'].copy()


# 修改优化函数，返回Y染色体浓度的平均值而不是效用值
def fit_copula_and_optimize_y_concentration(df_group):
    data = df_group[['孕妇BMI', '孕周', 'Y染色体浓度']].dropna()
    if len(data) < 30:
        print("数据不足，跳过")
        return None, None, None, None

    # Copula 拟合
    copula = GaussianMultivariate()
    copula.fit(data)

    # 从拟合的 copula 里采样大规模数据
    samples = copula.sample(5000)

    weeks = np.arange(10, 25.5, 0.5)
    y_concentrations = []  # 改为存储Y染色体浓度
    probabilities = []

    for t in weeks:
        bmi_mean = data['孕妇BMI'].mean()
        # 取条件子集 (孕周接近 t, BMI 接近均值)
        cond = samples[
            (np.abs(samples['孕周'] - t) <= 0.3) &
            (np.abs(samples['孕妇BMI'] - bmi_mean) <= 1.0)
            ]

        if len(cond) == 0:
            y_mean = 0
            prob = 0
        else:
            y_mean = np.mean(cond['Y染色体浓度'])  # 计算平均浓度
            prob = np.mean(cond['Y染色体浓度'] >= 0.04)

        y_concentrations.append(y_mean)
        probabilities.append(prob)

    # 找到Y浓度最高的时点（或者您可以根据需要调整选择标准）
    best_idx = int(np.argmax(y_concentrations))
    best_t = weeks[best_idx]

    return best_t, y_concentrations, probabilities, samples


# 运行修改后的函数
best_low, y_low, prob_low, samples_low = fit_copula_and_optimize_y_concentration(df_low)
best_mid, y_mid, prob_mid, samples_mid = fit_copula_and_optimize_y_concentration(df_mid)
best_high, y_high, prob_high, samples_high = fit_copula_and_optimize_y_concentration(df_high)

# 然后修改可视化代码
# 1. Y染色体浓度曲线对比
ax1 = fig.add_subplot(gs[0, :])
if y_low is not None:
    ax1.plot(weeks, y_low, label=f'Low BMI (Peak: {best_low:.1f} weeks)', linewidth=2.5)
if y_mid is not None:
    ax1.plot(weeks, y_mid, label=f'Mid BMI (Peak: {best_mid:.1f} weeks)', linewidth=2.5)
if y_high is not None:
    ax1.plot(weeks, y_high, label=f'High BMI (Peak: {best_high:.1f} weeks)', linewidth=2.5)

# 标记最佳时点（Y浓度最高点）
if best_low is not None:
    ax1.axvline(x=best_low, color='blue', linestyle='--', alpha=0.7)
if best_mid is not None:
    ax1.axvline(x=best_mid, color='green', linestyle='--', alpha=0.7)
if best_high is not None:
    ax1.axvline(x=best_high, color='red', linestyle='--', alpha=0.7)

# 标记阈值线
ax1.axhline(y=0.04, color='black', linestyle='--', linewidth=2, label='4% Threshold')

# 标记风险区域（保持不变）
ax1.axvspan(10, 12, alpha=0.1, color='green', label='Low Risk (≤12 weeks)')
ax1.axvspan(12, 27, alpha=0.1, color='yellow', label='Medium Risk (13-27 weeks)')
ax1.axvspan(27, 25.5, alpha=0.1, color='red', label='High Risk (≥28 weeks)')

ax1.set_xlabel('Gestational Week')
ax1.set_ylabel('Y Chromosome Concentration')
ax1.set_title('Y Chromosome Concentration by BMI Group and Gestational Week')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.show()
print('ok')
