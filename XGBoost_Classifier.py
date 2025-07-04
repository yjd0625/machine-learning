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
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier, plot_importance
import optuna

def objective(trial):
    # 加载数据
    data = datasets.load_diabetes()
    X = data.data
    y = (data.target > 100).astype(int)    # 将问题转化为两分类问题
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 定义超参数搜索空间
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0, 1),
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    # 训练XGBoost模型
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)

    # 交叉验证评估模型性能
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    accuracy = scores.mean()

    return accuracy

def main():
    parser = argparse.ArgumentParser(description="XGBoost分类器与Optuna超参数优化")
    parser.add_argument('--n_trials', type=int, default=10, help='Optuna优化的试验次数')
    args = parser.parse_args()

    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    # 输出最佳超参数组合
    print("Best trial:")
    trial = study.best_trial
    print("  Accuracy: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 使用最佳超参数重新训练模型
    best_params = trial.params
    best_params['objective'] = 'binary:logistic'
    best_params['use_label_encoder'] = False
    best_params['eval_metric'] = 'logloss'

    # 加载数据
    data = datasets.load_diabetes()
    X = data.data
    y = (data.target > 100).astype(int)  # 将问题转化为两分类问题
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # 评估模型
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(best_model, X, y, cv=kf, scoring='accuracy')
    print(f"Cross-validation scores: {scores}")
    print(f"Mean cross-validation score: {scores.mean()}")
    y_pred = best_model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 绘制ROC曲线
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # 特征重要性可视化
    plot_importance(best_model)
    plt.title('Feature Importance')
    plt.show()

if __name__ == '__main__':
    main()