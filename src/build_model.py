from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def split_data(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def train_model(x_train, y_train, model_type = "logreg", **kwargs):
    if model_type=="logreg":
        default_params = {"max_iter": 2000, "class_weight": "balanced"}
        default_params.update(kwargs)
        model = LogisticRegression(**default_params)
    elif model_type=="xgb":
        if "scale_pos_weight" not in kwargs:
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            kwargs["scale_pos_weight"] = n_neg / n_pos
       
        kwargs.setdefault("eval_metric", "logloss")
        kwargs.setdefault("use_label_encoder", False)

        model = XGBClassifier(**kwargs)
    elif model_type=="catboost":
        from catboost import CatBoostClassifier
        # separate categorical data columns
        cat_features = x_train.select_dtypes(include="object").columns.tolist()
    
        model = CatBoostClassifier(**kwargs)
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")
    if model_type=="catboost":
        model.fit(x_train, y_train, cat_features=cat_features)
    else:
        model.fit(x_train, y_train)

    return model

def predict(model, x):
    return model.predict(x)
