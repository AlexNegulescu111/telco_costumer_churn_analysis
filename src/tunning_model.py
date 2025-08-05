def tune_logistic_params(x_train, y_train, x_val, y_val, n_trials=50):
    import optuna
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
            "max_iter": 1000,
            "class_weight": "balanced"
        }

        model = LogisticRegression(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        return f1_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

def tune_xgb_params(x_train, y_train, x_val, y_val, n_trials=50):
    import optuna
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "eval_metric": "logloss",
            "use_label_encoder": False
        }

        model = XGBClassifier(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        return f1_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

def tune_cat_params(x_train, y_train, x_val, y_val, n_trials=50):
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.metrics import f1_score

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "loss_function": "Logloss",
            "eval_metric": "F1",
            "verbose": 0
        }

        cat_features = x_train.select_dtypes(include="object").columns.tolist()
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train, cat_features=cat_features)
        y_pred = model.predict(x_val)
        return f1_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params