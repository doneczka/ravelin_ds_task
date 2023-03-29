import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
import ipaddress
import geoip2.database
import socket

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    MeanMedianImputer,
    CategoricalImputer,
    AddMissingIndicator,
)
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection
from feature_engine.outliers import Winsorizer
from boruta import BorutaPy
import xgboost as xg
from sklearn.metrics import classification_report, confusion_matrix

from utils import *


pd.set_option("display.max_colwidth", 0)
sns.set_style("white")
sns.set(rc={"figure.figsize": (8, 6)})
custom_palette = {
    False: sns.color_palette("deep")[2],
    True: sns.color_palette("deep")[3],
}

plt.rcParams.update({"axes.titlesize": 14})


def extract_values(data, column_name):
    unique_keys = data[column_name][0][0]
    for key in unique_keys:
        data[f"{column_name}_{key}"] = data[column_name].apply(
            lambda x: [d[key] for d in x if key in d]
        )


def is_valid_domain(domain):
    try:
        socket.gethostbyname(domain)
        return True
    except socket.error:
        return False


def is_proper_ip(address):
    try:
        ip = ipaddress.ip_address(address)
        if ip.is_private:
            return "local"
        else:
            return "global"
    except ValueError:
        return "invalid"


def get_ip_localization(row, reader):
    try:
        response = reader.country(row["customerIPAddress"])
        return response.country.name
    except geoip2.errors.AddressNotFoundError:
        return "not_found"


def extract_address_info(address_str):
    lines = address_str.split("\n")
    address_line = lines[-1].strip()

    address_parts = address_line.split(",")
    city = address_parts[0]

    if len(address_parts) > 1:
        state_zip = address_parts[1].strip().split(" ")
        state = state_zip[0]
        zip_code = state_zip[1] if len(state_zip) > 1 else ""
    else:
        state = ""
        zip_code = ""

    return city, state, zip_code


def get_state_name(state_abbrev, states_df):
    if state_abbrev in states_df.index:
        return states_df.loc[state_abbrev, "State"]
    else:
        return "not_present"


def is_billing_address_in_shipping_address(row):
    billing_address = row["customerBillingAddress"]
    shipping_address = row["orders_orderShippingAddress"]
    return billing_address in shipping_address


def count_unique(lst):
    unique_numbers = set()
    for string in lst:
        match = re.search(r"\d{5}-\d{4}|\d{5}$", string)
        if match:
            unique_numbers.add(match.group(0))
    return len(unique_numbers)


def highlight_fraudulent_row(row):
    color = "indianred" if row["fraudulent"] else ""
    return [f"background-color: {color}" for val in row]


def create_countplot_with_hue(data, x_column, order=None):
    sns.countplot(
        data=data, x=x_column, hue="fraudulent", palette=custom_palette, order=order
    )
    plt.title(f'Fraudulent Status by {x_column.replace("_", " ").title()}')


def compare_lists(list1, list2):
    return list1 == list2


def histplot_with_hue(data, x_column):
    sns.histplot(
        data=data,
        x=x_column,
        hue="fraudulent",
        palette=custom_palette,
    )
    plt.title(f"Fraudulent status by {x_column.replace('_', ' ').title()}")


def countplot_one_cat(data, x_column, title):
    sns.countplot(
        data=data,
        x=x_column,
        palette=custom_palette,
    )
    plt.title(f"Fraudulent Status for {title}")


def normalized_cm(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    normalized_cm = np.round((cm / np.sum(cm, axis=1).reshape(-1, 1)), 3)
    sns.heatmap(normalized_cm, annot=True, cmap="Greens")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.title(f"Normalized confusion matrix for {model_name}")


def create_heatmap(df, title, figsize):
    corr_matrix = df.iloc[:, 2:].corr(method="pearson")
    sns.set(font_scale=1)
    f, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="YlGnBu",
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.5},
    )
    f.tight_layout()
    ax.set_title(
        f"Correlation heatmap of {title}",
        fontdict={"fontsize": 16},
        pad=12,
    )
    plt.xlabel("")
    plt.ylabel("")


def build_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    normalized_cm(y_test, pred, {model})
    return model
