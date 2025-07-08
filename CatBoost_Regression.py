"""
==================================================
Author：Yao Jiadong
e-mail：jdyao0625@foxmail.com
GitHub：https://github.com/yjd0625/machine-learning.git
create：2025-07-08
update：2025-07-08
version：1.0
==================================================
"""

import argparse
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
import optuna

def objective(trial):
    # 加载数据
    data = datasets.load_diabetes()
    X = data.data
    y = data.target  # 回归任务的目标变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 定义超参数搜索空间
    param = {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'verbose': 0,
        'random_state': 42,
        'logging_level': 'Silent'  # 禁用日志输出
    }

    # 训练 CatBoost 回归模型
    model = CatBoostRegressor(**param)
    model.fit(X_train, y_train)

    # 交叉验证评估模型性能
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mse = -scores.mean()  # 负均方误差

    return mse

def main():
    parser = argparse.ArgumentParser(description="CatBoost回归器与Optuna超参数优化")
    parser.add_argument('--n_trials', type=int, default=1, help='Optuna优化的试验次数')
    args = parser.parse_args()

    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)

    # 输出最佳超参数组合
    print("Best trial:")
    trial = study.best_trial
    print("  MSE: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 使用最佳超参数重新训练模型
    best_params = trial.params

    # 加载数据
    data = datasets.load_diabetes()
    X = data.data
    y = data.target  # 回归任务的目标变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    best_params['logging_level'] = 'Silent'
    best_model = CatBoostRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # 评估模型
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(best_model, X, y, cv=kf, scoring='neg_mean_squared_error')
    mse_scores = -scores
    print(f"Cross-validation MSE scores: {mse_scores}")
    print(f"Mean cross-validation MSE score: {mse_scores.mean()}")
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test R² Score: {r2}")

    # 绘制实际值与预测值的对比图
    plt.figure()
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ideal Line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()