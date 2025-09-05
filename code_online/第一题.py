import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体，优先使用SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

male = pd.read_csv(r"C:\Users\86137\Desktop\国赛\男胎儿.csv")
print(male.info())
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

##然后处理男胎儿
male.drop(["序号","孕妇代码","末次月经","检测日期","染色体的非整倍体","胎儿是否健康"],axis=1,inplace=True)
male['孕周数值'] = male['检测孕周'].apply(convert_gestational_week)
male.drop("检测孕周",axis=1,inplace=True)
male_one = pd.get_dummies(male,dtype='int8')
print(male_one.head())

# 计算子图的行列数
n_features = len(male_one.columns)
n_cols = 5  # 每行显示3个子图
n_rows = (n_features + n_cols - 1) // n_cols  # 计算需要的行数

# 创建子图
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()  # 将axes数组展平，便于迭代

# 为每个特征绘制QQ图
for i, column in enumerate(male_one.columns):
    # 使用statsmodels绘制QQ图
    sm.qqplot(male_one[column].dropna(), line='s', ax=axes[i])
    axes[i].set_title(f'{column}的QQ图')
    axes[i].set_xlabel('理论分位数')
    axes[i].set_ylabel('样本分位数')

# 隐藏多余的子图
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# 可选：同时输出每个特征的正态性检验统计结果
print("各特征正态性检验结果（Shapiro-Wilk检验）:")
print("特征名\t\tW统计量\t\tp值\t\t是否正态（α=0.05）")
print("-" * 65)
for column in male_one.columns:
    stat, p_value = stats.shapiro(male_one[column])
    is_normal = "是" if p_value > 0.05 else "否"
    print(f"{column}\t\t{stat:.4f}\t\t{p_value:.4e}\t\t{is_normal}")


def plot_spearman_heatmap(data):
    # 计算斯皮尔曼相关系数
    corr_matrix = data.corr(method='spearman')

    # 创建热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                linewidths=0.5,
                vmin=-1, vmax=1,
                annot_kws={"size": 8})

    plt.title('spearman matrix', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
plot_spearman_heatmap(male_one)

# 计算相关系数
def calculate_correlations(df, target_col='Y染色体浓度'):
    # 分离特征和目标变量
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # 计算皮尔逊相关系数
    pearson_corr = {}
    for col in X.columns:
        corr, p_value = stats.pearsonr(X[col], y)
        pearson_corr[col] = {'相关系数': corr, 'p值': p_value}
    # 计算斯皮尔曼相关系数
    spearman_corr = {}
    for col in X.columns:
        corr, p_value = stats.spearmanr(X[col], y)
        spearman_corr[col] = {'相关系数': corr, 'p值': p_value}
    # 创建相关系数DataFrame
    pearson_df = pd.DataFrame(pearson_corr).T
    pearson_df.columns = ['皮尔逊相关系数', '皮尔逊p值']
    spearman_df = pd.DataFrame(spearman_corr).T
    spearman_df.columns = ['斯皮尔曼相关系数', '斯皮尔曼p值']
    # 合并结果
    correlation_df = pd.concat([pearson_df, spearman_df], axis=1)
    correlation_df = correlation_df.sort_values('皮尔逊相关系数', key=abs, ascending=False)
    return correlation_df
correlation_df = calculate_correlations(male_one)
print(correlation_df)

X = male_one.drop(columns=['Y染色体浓度'])
y = male_one['Y染色体浓度']
# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 使用测试集进行预测
y_pred = model.predict(X)

# 计算评估指标
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# 输出模型参数和评估结果
print("=== 线性回归模型结果 ===")
print(f"特征系数 (Coefficients): {model.coef_}")
print(f"截距 (Intercept): {model.intercept_}")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")


def lasso_regression_analysis(df, target_col='Y染色体浓度'):
    # 分离特征和目标变量
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 使用交叉验证选择最佳alpha值
    lasso_cv = LassoCV(cv=5, random_state=42)
    lasso_cv.fit(X, y)

    # 使用最佳alpha值拟合LASSO模型
    best_alpha = lasso_cv.alpha_
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X, y)

    # 获取系数
    coefficients = pd.DataFrame({
        '特征': X.columns,
        '系数': lasso.coef_,
        '截距': lasso.intercept_
    })
    coefficients = coefficients.sort_values('系数', key=abs, ascending=False)

    # 计算模型性能
    y_pred = lasso.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return {
        '最佳alpha': best_alpha,
        '系数': coefficients,
        'R²': r2,
        'MSE': mse,
        '模型': lasso
    }

lasso_results = lasso_regression_analysis(male_one)
print("\nLASSO回归分析结果:")
print(f"最佳alpha值: {lasso_results['最佳alpha']}")
print(f"模型R²: {lasso_results['R²']:.4f}")
print(f"模型MSE: {lasso_results['MSE']:.4f}")
print("特征系数:")
print(lasso_results['系数'])


def polynomial_regression_analysis(df, target_col='Y染色体浓度', degree=2):
    # 分离特征和目标变量
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 选择与目标变量相关性最高的几个特征
    corr_with_target = df.corr()[target_col].abs().sort_values(ascending=False)
    top_features = corr_with_target.index[1:6]  # 选择前5个最相关的特征（排除目标变量自身）
    # 创建多项式特征管道
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    # 拟合模型
    poly_model.fit(X, y)

    # 计算模型性能
    y_pred = poly_model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # 获取多项式特征名称
    poly_features = PolynomialFeatures(degree=degree).fit(X)
    feature_names = poly_features.get_feature_names_out(X.columns)
    coefficients = poly_model.named_steps['linear'].coef_
    intercept = poly_model.named_steps['linear'].intercept_
    coefficients = pd.DataFrame({
        '特征': feature_names,
        '系数': coefficients
    })
    coefficients = coefficients.sort_values('系数', key=abs, ascending=False)
    return {
        '选定的特征': top_features.tolist(),
        'R²': r2,
        'MSE': mse,
        '模型': poly_model,
        '特征名称': feature_names,
        '系数': coefficients,
        '截距': intercept
    }

poly_results = polynomial_regression_analysis(male_one, degree=2)
print("\n多项式回归分析结果:")
print(f"选定的特征: {poly_results['选定的特征']}")
print(f"模型R²: {poly_results['R²']:.4f}")
print(f"模型MSE: {poly_results['MSE']:.4f}")
print(poly_results['系数'])
print(poly_results['截距'])