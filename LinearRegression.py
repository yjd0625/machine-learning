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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


class AlgorithmTemplate:
    """
    [算法名称] 实现类

    参数说明：
    - param1: 参数1说明 (默认值)
    - param2: 参数2说明 (默认值)
    """

    def __init__(self, param1=0.1, param2=100):
        self.param1 = param1
        self.param2 = param2
        self.weights = None
        # 添加其他初始化参数

    def repreprocess(self, X):
        # 实现标准化、归一化等预处理
        # 示例：标准化处理
        if not hasattr(self, 'mean'):
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
        return (X - self.mean) / (self.std + 1e-8)

    def fit(self, X, y, epochs=100):
        # 1. 预处理数据
        X_processed = self._preprocess_data(X)

        # 2. 初始化权重
        n_features = X_processed.shape[1]
        self.weights = np.zeros(n_features)

        # 3. 训练循环
        for epoch in range(epochs):
            # 实现核心训练逻辑
            # 示例：梯度下降
            predictions = np.dot(X_processed, self.weights)
            errors = y - predictions

            # 更新权重
            gradient = -2 * np.dot(X_processed.T, errors) / len(y)
            self.weights -= self.param1 * gradient

            # 打印训练进度
            if epoch % 10 == 0:
                loss = np.mean(errors ** 2)
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        X_processed = self._preprocess_data(X)
        return np.dot(X_processed, self.weights)

    def evaluate(self, X, y, metric='mse'):
        preds = self.predict(X)
        if metric == 'mse':
            return mean_squared_error(y, preds)
        elif metric == 'acc':
            # 分类问题示例
            pred_labels = (preds > 0.5).astype(int)
            return accuracy_score(y, pred_labels)
        else:
            raise ValueError(f"Unsupported metric: {metric}")


def load_sample_data():
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
    return X, y


def visualize_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.grid(True)
    plt.savefig('results.png', dpi=300)
    plt.show()


# =============== 主程序/示例用法 ===============
if __name__ == "__main__":
    print("\n=== [算法名称] 实现示例 ===")

    # 1. 加载数据
    X, y = load_sample_data()
    print(f"数据集形状: X={X.shape}, y={y.shape}")

    # 2. 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. 初始化模型
    model = AlgorithmTemplate(param1=0.01, param2=50)

    # 4. 训练模型
    print("\n开始训练...")
    model.fit(X_train, y_train, epochs=100)

    # 5. 评估模型
    train_mse = model.evaluate(X_train, y_train)
    test_mse = model.evaluate(X_test, y_test)
    print(f"\n训练MSE: {train_mse:.4f}, 测试MSE: {test_mse:.4f}")

    # 6. 可视化结果
    y_pred = model.predict(X_test)
    visualize_results(y_test, y_pred)

    print("\n=== 执行完成 ===")