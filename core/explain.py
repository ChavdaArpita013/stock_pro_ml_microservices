import shap
import joblib
import os
import pandas as pd
from utils.config import MODEL_DIR

def get_explaination(latest: list , model_name: str):

    model_path = os.path.join(MODEL_DIR , f"{model_name}.pkl")
    model = joblib.load(model_path)

    explainer = shap.Explainer(model , latest)
    shap_value = explainer(latest)

    explanation = dict(zip(latest.columns , shap_value.values[0]))
    sorted_explanation = dict(sorted(explanation.items() , key=lambda item: abs(item[1]), reverse=True))

    return{
        "model" : model_name,
        "most_influential_factors": sorted_explanation
    }
