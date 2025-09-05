import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据读取
male_df = pd.read_excel("ref.xlsx", sheet_name="男胎检测数据")

# 数据预处理
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
male_df["BMI"] = male_df["孕妇BMI"]
male_clean = male_df[["孕周数","BMI","Y染色体浓度"]].dropna()

X = male_clean[["孕周数", "BMI"]]
y = male_clean["Y染色体浓度"]
x1 = male_clean["孕周数"]
x2 = male_clean["BMI"]


# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

