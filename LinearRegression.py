"""
==================================================
作者：Yao Jiadong
邮箱：jdyao0625@foxmail.com
GitHub：https://github.com/yjd0625/machine-learning.git
创建日期：2025-07-03
最后修改：2025-07-03
版本：1.0
==================================================
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


parser = argparse.ArgumentParser(description='线性回归模型')
parser.add_argument('--n_samples', type=int, default=100, help='样本数量')
parser.add_argument('--n_features', type=int, default=1, help='特征数量')
parser.add_argument('--noise', type=float, default=10, help='噪声水平')
parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
parser.add_argument('--alpha', type=float, default=0.0, help='正则化参数（岭回归）')
parser.add_argument('--cv', type=int, default=5, help='K折交叉验证折数')
parser.add_argument('--visualize', type=bool, default=True, help='是否可视化')
args = parser.parse_args()

def main():
    # 生成模拟数据，划分训练和测试集
    X, y = make_regression(n_samples=args.n_samples, n_features=args.n_features, noise=args.noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    # 根据正则化参数选择模型
    if args.alpha > 0:
        model = Ridge(alpha=args.alpha)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)

    # 计算测试集均方误差
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集均方误差（MSE）: {mse:.4f}")

    # 使用交叉验证评估模型性能
    cv_scores = cross_val_score(model, X, y, cv=args.cv, scoring='neg_mean_squared_error')
    cv_mse = -np.mean(cv_scores)
    print(f"{args.cv}折交叉验证均方误差（MSE）: {cv_mse:.4f}")

    # 可视化数据和拟合的直线
    if args.visualize:
        # 绘制训练数据点和测试数据点
        plt.scatter(X_train, y_train, color='blue', label='训练数据')
        plt.scatter(X_test, y_test, color='green', label='测试数据')

        # 绘制拟合的直线
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        plt.plot(x_line, y_line, color='red', label='拟合直线')

        plt.xlabel('特征')
        plt.ylabel('目标')
        plt.title('线性回归拟合结果')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()