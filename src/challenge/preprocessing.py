import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def build_new_features(df_raw):
    df = df_raw.copy()
    df["sin_wd"] = df["wd"].apply(convert_degree_to_radian)
    df["hour"] = df.index.map(get_hour)
    df["day"] = df.index.map(get_day)
    df["month"] = df.index.map(get_month)
    return df.drop("wd", axis=1)
    

def convert_degree_to_radian(degree: float) -> float:
    radian = degree / 180
    return np.sin(radian * np.pi)

def get_hour(datetime) -> int:
    return datetime.hour

def get_day(datetime) -> int:
    return datetime.day

def get_month(datetime) -> int:
    return datetime.month

def build_polynomial_wind_features(
    polynomial_features: PolynomialFeatures,
    data,
    selected_wind_features,
    selected_time_features,
):
    polynomial_data = polynomial_features.fit_transform(data[selected_wind_features])
    df_polynomial_data = pd.DataFrame(
        polynomial_data,
        columns=get_polynomial_names(polynomial_features, selected_wind_features),
        index=data.index
    )
    return pd.concat([df_polynomial_data, data[selected_time_features]], axis=1)


def get_polynomial_names(polynomial_features, original_names):
    mapped_names = [f"x{_id}" for _id in range(len(original_names))]
    polynomial_feature_names = polynomial_features.get_feature_names()
    
    polynomial_names = polynomial_feature_names
    for original_name, mapped_name in zip(original_names, mapped_names):
        polynomial_names = replace_in_list(polynomial_names, mapped_name, original_name)
    return polynomial_names


def replace_in_list(names, original, substitute):
    return [char.replace(original, substitute) for char in names]
