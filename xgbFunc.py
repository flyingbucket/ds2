import json
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import optuna
from optuna.trial import TrialState
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def encode_features(df: pd.DataFrame, fill_value="__missing__"):
    df = df.copy()  # 避免修改原始数据
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        print("没有非数值类型的列，无需编码。")
        return df

    # 填充缺失值
    df[categorical_cols] = df[categorical_cols].fillna(fill_value)

    # 编码器
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    # 拟合并替换原列
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    return df


def xgb_reg_search(X_train, y_train):
    param_dist = {
        "n_estimators": randint(100, 300),  # 树的数量
        "learning_rate": uniform(0.01, 0.3),  # 学习率
        "max_depth": randint(3, 10),  # 树的最大深度
        "min_child_weight": randint(1, 10),  # 子节点所需最小样本数
        "gamma": uniform(0, 0.5),  # 最小损失减少量
        "subsample": uniform(0.7, 0.3),  # 样本采样比例
        "colsample_bytree": uniform(0.7, 0.3),  # 特征采样比例
        "scale_pos_weight": uniform(0.5, 3),  # 类别不平衡的加权因子
        "alpha": uniform(0, 1),  # L1正则化项
        "lambda": uniform(0, 1),  # L2正则化项
    }

    model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=True,
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print("Best Score:", best_score)
    return best_params, best_score


def xgb_clf_search_gpu_0(X_train, y_train, n_iter=100):
    # 自动编码字符串标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    num_classes = len(np.unique(y_encoded))

    param_dist = {
        "n_estimators": randint(100, 300),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 10),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 0.5),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 1),
    }

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",  # 使用 GPU 训练
        device="cuda",
        n_jobs=-1,
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=True,
    )

    random_search.fit(X_train, y_encoded)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print("Best Accuracy:", best_score)

    return best_params, best_score, label_encoder, num_classes


def xgb_clf_search(X_train, y_train, n_iter=100):
    # 自动编码字符串标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    num_classes = len(np.unique(y_encoded))

    param_dist = {
        "n_estimators": randint(100, 300),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 10),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 0.5),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "reg_alpha": uniform(0, 1),  # 注意参数名不同
        "reg_lambda": uniform(0, 1),
    }

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_jobs=-1,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        device="gpu",
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=True,
    )

    random_search.fit(X_train, y_encoded)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print("Best Accuracy:", best_score)

    return best_params, best_score, label_encoder, num_classes


def evaluate_multiclass(y_test, y_pred, label_encoder=None):
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # 构造带标签的混淆矩阵
    if label_encoder is not None:
        labels = label_encoder.classes_
    else:
        labels = sorted(
            list(set(y_test) | set(y_pred))
        )  # 如果没传入编码器，也能正常运行

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_dict = cm_df.to_dict(orient="index")

    # 组织输出结果
    result = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "confusion_matrix": cm_dict,
    }

    # 打印输出
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(cm_df)

    return result


def evaluate_multiclass0(y_test, y_pred):
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    # 计算精确度
    precision = precision_score(y_test, y_pred, average="weighted")  # 使用加权平均
    # 计算召回率
    recall = recall_score(y_test, y_pred, average="weighted")  # 使用加权平均
    # 计算F1分数
    f1 = f1_score(y_test, y_pred, average="weighted")  # 使用加权平均
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    result = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "confusion_matrix": conf_matrix,
    }
    # 打印结果
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    return result


def evaluate_reg(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)
    result = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    return result


def visualize_importance(model):
    figs = []

    # Feature Importance - weight
    fig1, ax1 = plt.subplots()
    fig1, ax1 = plt.subplots(figsize=(15, 15))
    fig1.subplots_adjust(left=0.18)
    xgb.plot_importance(model, importance_type="weight", ax=ax1)
    ax1.set_title("Feature Importance (Weight)")
    figs.append((fig1, "weight"))

    # Feature Importance - gain
    fig2, ax2 = plt.subplots()
    fig2, ax2 = plt.subplots(figsize=(15, 15))
    fig2.subplots_adjust(left=0.18)
    xgb.plot_importance(model, importance_type="gain", ax=ax2)
    ax2.set_title("Feature Importance (Gain)")
    figs.append((fig2, "gain"))

    # Feature Importance - cover
    fig3, ax3 = plt.subplots()
    fig3, ax3 = plt.subplots(figsize=(15, 15))
    fig3.subplots_adjust(left=0.18)
    xgb.plot_importance(model, importance_type="cover", ax=ax3)
    ax3.set_title("Feature Importance (Cover)")
    figs.append((fig3, "cover"))
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    return figs


def visualize_multiclass_shap0(model, X_train, sample_size=10000):
    booster = model if isinstance(model, xgb.Booster) else model.get_booster()
    if len(X_train) > sample_size:
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
    else:
        X_sample = X_train

    explainer = shap.TreeExplainer(booster, X_sample)
    shap_values = explainer.shap_values(X_sample)  # List[ndarray] per class

    figs = []
    for class_idx, shap_class_vals in enumerate(shap_values):
        data_for_plot = (
            X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
        )
        shap.summary_plot(
            shap_class_vals,
            data_for_plot,
            feature_names=np.array(X_sample.columns),
            show=False,
            plot_type="dot",
        )
        fig = plt.gcf()
        fig.set_size_inches(10, 12)
        plt.title(f"SHAP Summary Plot - Class {class_idx}")
        figs.append(fig)
        plt.clf()
    return figs


def visualize_multiclass_shap(model, X_train, sample_size=1000):
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    import xgboost as xgb

    booster = model if isinstance(model, xgb.Booster) else model.get_booster()

    # 采样
    if len(X_train) > sample_size:
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
    else:
        X_sample = X_train

    explainer = shap.TreeExplainer(booster, X_train)
    shap_values = explainer.shap_values(X_sample)
    print(f"X_sample.shap: {X_sample.shape}")
    print(f"shap_values.shape:{shap_values.shape}")
    figs = []

    for i in range(shap_values.shape[2]):  # 遍历类别 (n_classes)
        # 取出每个类别对应的 SHAP 值，形状应为 [n_samples, n_features]
        class_shap_values = shap_values[:, :, i]
        print(f"class_shap_values.shape:{class_shap_values.shape}")

        plt.figure()
        shap.summary_plot(
            class_shap_values,
            X_sample,
            feature_names=X_train.columns,
            show=False,
            plot_type="dot",
        )
        fig = plt.gcf()
        fig.set_size_inches(10, 12)
        plt.title(f"SHAP Summary Plot - Class {i}")
        figs.append(fig)
        plt.close(fig)

    return figs


def visualize_reg_bin_shap(model, X_train, sample_size=10000):
    booster = model if isinstance(model, xgb.Booster) else model.get_booster()
    if len(X_train) > sample_size:
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
    else:
        X_sample = X_train
    # 关闭 additivity check
    explainer = shap.TreeExplainer(booster, X_train)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=np.array(X_train.columns),
        show=False,
        plot_type="dot",
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 12)
    plt.title("SHAP Summary Plot")
    return [
        fig
    ]  # 注意返回是列表，与你的 draw_shap 中的 `for i, fig in enumerate(figs)` 保持一致


def visualize_reg_shap(model, X_train, sample_size=10000):
    booster = model if isinstance(model, xgb.Booster) else model.get_booster()
    if len(X_train) > sample_size:
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
    else:
        X_sample = X_train
    explainer = shap.TreeExplainer(booster, X_train)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=np.array(X_train.columns),
        show=False,
        plot_type="dot",
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 12)
    plt.title("SHAP Summary Plot")
    return fig


def xgb_clf_search_gpu(X, y, n_iter=100, cv=3, random_state=42):
    rng = np.random.RandomState(random_state)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # 创建 GPU DMatrix 数据，一次性复制到 GPU
    dmatrix = xgb.DMatrix(X, label=y_encoded)

    # 参数空间
    param_dist = {
        "learning_rate": lambda: uniform(0.01, 0.3).rvs(random_state=rng),
        "max_depth": lambda: randint(3, 10).rvs(random_state=rng),
        "min_child_weight": lambda: randint(1, 10).rvs(random_state=rng),
        "gamma": lambda: uniform(0, 0.5).rvs(random_state=rng),
        "subsample": lambda: uniform(0.7, 0.3).rvs(random_state=rng),
        "colsample_bytree": lambda: uniform(0.7, 0.3).rvs(random_state=rng),
        "reg_alpha": lambda: uniform(0, 1).rvs(random_state=rng),
        "reg_lambda": lambda: uniform(0, 1).rvs(random_state=rng),
        "n_estimators": lambda: randint(100, 300).rvs(random_state=rng),
    }

    # K折交叉验证划分
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    best_score = 0
    best_params = None

    print("Starting random search...")

    for i in tqdm(range(n_iter)):
        # 随机采样一组参数
        sampled_params = {k: sampler() for k, sampler in param_dist.items()}

        # 设置 XGBoost 参数
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "tree_method": "gpu_hist",
            "device": "cuda",
            "learning_rate": sampled_params["learning_rate"],
            "max_depth": int(sampled_params["max_depth"]),
            "min_child_weight": sampled_params["min_child_weight"],
            "gamma": sampled_params["gamma"],
            "subsample": sampled_params["subsample"],
            "colsample_bytree": sampled_params["colsample_bytree"],
            "reg_alpha": sampled_params["reg_alpha"],
            "reg_lambda": sampled_params["reg_lambda"],
        }

        n_estimators = int(sampled_params["n_estimators"])
        scores = []

        # 手动做 K 折交叉验证
        for train_idx, valid_idx in skf.split(X, y_encoded):
            dtrain = dmatrix.slice(train_idx)
            dvalid = dmatrix.slice(valid_idx)

            evals = [(dvalid, "eval")]
            booster = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=n_estimators,
                evals=evals,
                verbose_eval=False,
            )

            preds = booster.predict(dvalid)
            # print(np.unique(preds))
            acc = (preds.argmax(axis=1) == y_encoded[valid_idx]).mean()
            scores.append(acc)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = sampled_params

    print("\nBest Parameters:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print("Best Accuracy:", best_score)

    return best_params, best_score, label_encoder, num_classes


def xgb_binary_search_optuna(
    X, y, result_path, n_trials=100, cv=3, random_state=42, search_space=None
):
    """
    使用 Optuna 进行超参数优化的 XGBoost 二分类器训练。
    """
    rng = np.random.RandomState(random_state)

    # 标签编码（二分类）
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # K折交叉验证
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def save_best_params(study, result_path):
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete_trials:
            return
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        os.makedirs(result_path, exist_ok=True)
        with open(os.path.join(result_path, "current_best_params.json"), "w") as f:
            json.dump({"params": best_params, "score": best_score}, f, indent=4)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "gpu_hist",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "subsample": trial.suggest_float("subsample", 0.7, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        }
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)

        if search_space:
            for key, value in search_space.items():
                if key in params:
                    params[key] = value

        sample_weights = compute_sample_weight("balanced", y_encoded)
        dmatrix = xgb.DMatrix(X, label=y_encoded, weight=sample_weights)

        scores = []
        for train_idx, valid_idx in skf.split(X, y_encoded):
            dtrain = dmatrix.slice(train_idx)
            dvalid = dmatrix.slice(valid_idx)

            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dvalid, "eval")],
                verbose_eval=False,
            )

            preds = booster.predict(dvalid)
            pred_labels = (preds > 0.5).astype(int)
            acc = accuracy_score(y_encoded[valid_idx], pred_labels)
            scores.append(acc)

        save_best_params(study, result_path)
        print("Saved current best parameters.")
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Accuracy: {best_score}")

    return best_params, best_score, label_encoder


def xgb_clf_search_optuna(
    X, y, result_path, n_trials=100, cv=3, random_state=42, search_space=None
):
    """
    使用 Optuna 进行超参数优化的 XGBoost 分类器训练。
    """
    rng = np.random.RandomState(random_state)

    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # K折交叉验证划分
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def save_best_params(study, result_path):
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete_trials:
            return  # 没有成功的 trial，直接返回
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value

        # 保存为 JSON 文件
        with open(os.path.join(result_path, "current_best_params.json"), "w") as f:
            json.dump({"params": best_params, "score": best_score}, f, indent=4)

    # 定义 Optuna 优化目标函数
    def objective(trial):
        # 随机采样超参数
        params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "tree_method": "gpu_hist",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "subsample": trial.suggest_float("subsample", 0.7, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        }
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)

        if search_space:
            for key, value in search_space.items():
                if key in params:
                    params[key] = value
        # 计算样本权重，处理类别不平衡
        sample_weights = compute_sample_weight("balanced", y_encoded)

        # 创建 DMatrix
        dmatrix = xgb.DMatrix(X, label=y_encoded, weight=sample_weights)

        scores = []
        for train_idx, valid_idx in skf.split(X, y_encoded):
            dtrain = dmatrix.slice(train_idx)
            dvalid = dmatrix.slice(valid_idx)

            evals = [(dvalid, "eval")]

            # 训练 XGBoost 模型
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=evals,
                verbose_eval=False,
            )

            # 获取预测结果
            preds = booster.predict(dvalid)
            acc = (preds.argmax(axis=1) == y_encoded[valid_idx]).mean()
            scores.append(acc)
        save_best_params(study, result_path)
        print("saved current best params")
        # 返回平均准确率
        return np.mean(scores)

    # 创建 Optuna study 对象，使用 GPU 优化
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # 输出最优参数和最优结果
    best_params = study.best_params
    best_score = study.best_value
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Accuracy: {best_score}")

    return best_params, best_score, label_encoder, num_classes
