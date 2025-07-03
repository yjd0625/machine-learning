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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


parser = argparse.ArgumentParser(description="SVM分类器")
parser.add_argument('--kernel', type=str, default='linear', choices=['linear', 'poly', 'rbf', 'sigmoid'],
    help='SVM核函数类型')
parser.add_argument('--C', type=float, default=1.0, help='正则化参数')
parser.add_argument('--gamma', type=str, default='scale', choices=['scale', 'auto'],
    help='核函数的系数')
args = parser.parse_args()

def main():
    # 加载数据集
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 只取前两个特征方便可视化
    y = iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练模型
    model = SVC(kernel=args.kernel, C=args.C, gamma=args.gamma)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 绘制决策边界
    h = .02  # 网格的步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('SVM决策边界')
    plt.show()

if __name__ == '__main__':
    main()