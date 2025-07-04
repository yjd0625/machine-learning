"""
==================================================
Author：Yao Jiadong
e-mail：jdyao0625@foxmail.com
GitHub：https://github.com/yjd0625/machine-learning.git
create：2025-07-04
update：2025-07-04
version：1.0
==================================================
"""

import argparse
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
import optuna

def objective(trial):
    # 加载数据
    data = datasets.load_diabetes()
    X = data.data
    y = data.target  # 回归任务的目标变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 定义超参数搜索空间
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0, 1),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # 训练XGBoost回归模型
    model = XGBRegressor(**param)
    model.fit(X_train, y_train)

    # 交叉验证评估模型性能
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mse = -scores.mean()  # 负均方误差

    return mse

def main():
    parser = argparse.ArgumentParser(description="XGBoost回归器与Optuna超参数优化")
    parser.add_argument('--n_trials', type=int, default=10, help='Optuna优化的试验次数')
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
    best_params['objective'] = 'reg:squarederror'
    best_params['eval_metric'] = 'rmse'

    # 加载数据
    data = datasets.load_diabetes()
    X = data.data
    y = data.target  # 回归任务的目标变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    best_model = XGBRegressor(**best_params)
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

    # 特征重要性可视化
    plot_importance(best_model)
    plt.title('Feature Importance')
    plt.show()

if __name__ == '__main__':
    main()