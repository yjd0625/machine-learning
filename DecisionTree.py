import argparse
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser(description="决策树分类器或回归器")
    parser.add_argument('--mode', type=str, default='classification', choices=['classification', 'regression'],
                        help='选择模式：分类或回归')
    parser.add_argument('--criterion', type=str, default='gini', choices=['gini', 'entropy', 'mse', 'friedman_mse', 'mae'],
                        help='决策树的划分标准')
    parser.add_argument('--max_depth', type=int, default=None, help='决策树的最大深度')
    parser.add_argument('--min_samples_split', type=int, default=2, help='分裂内部节点所需的最小样本数')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='叶节点的最小样本数')
    args = parser.parse_args()

    # 加载数据
    if args.mode == 'classification':
        data = datasets.load_iris()
    elif args.mode == 'regression':
        data = datasets.load_boston()
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练决策树模型
    if args.mode == 'classification':
        model = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth,
                                       min_samples_split=args.min_samples_split,
                                       min_samples_leaf=args.min_samples_leaf)
    elif args.mode == 'regression':
        model = DecisionTreeRegressor(criterion=args.criterion, max_depth=args.max_depth,
                                      min_samples_split=args.min_samples_split,
                                      min_samples_leaf=args.min_samples_leaf)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    if args.mode == 'classification':
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    elif args.mode == 'regression':
        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("R^2 Score:", r2_score(y_test, y_pred))

    # 可视化决策树
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names if args.mode == 'classification' else None)
    plt.title("Decision Tree Visualization")
    plt.show()


if __name__ == '__main__':
    main()