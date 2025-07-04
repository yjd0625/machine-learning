"""
==================================================
Author：Yao Jiadong
e-mail：jdyao0625@foxmail.com
GitHub：https://github.com/yjd0625/machine-learning.git
create：2025-07-03
update：2025-07-03
version：1.0
==================================================
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser(description="逻辑回归分类器")
parser.add_argument('--solver', type=str, default='lbfgs', choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    help='优化算法')
parser.add_argument('--C', type=float, default=1.0, help='正则化强度的倒数')
parser.add_argument('--max_iter', type=int, default=100, help='最大迭代次数')
args = parser.parse_args()

def main():
    # 加载数据
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 只取前两个特征以便于可视化
    y = (iris.target != 0).astype(int)  # 将问题简化为二分类问题

    # 数据预处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression(solver=args.solver, C=args.C, max_iter=args.max_iter)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 绘制ROC曲线
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # 绘制决策边界
    h = .02  # 网格的步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

if __name__ == '__main__':
    main()