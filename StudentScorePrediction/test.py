# import yaml

# from src.data_loader import load_data
# from src.features import prepare_features

# with open("configs/config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# df = load_data(
#     config["data"]["db_path"],
#     config["data"]["table_name"]
# )

# X, y = prepare_features(
#     df,
#     target_col=config["data"]["target_col"],
#     drop_cols=config["data"]["drop_cols"]
# )

# print(X.shape)
# print(y.shape)
# print(X.head())

import yaml

from src.data_loader import load_data
from src.features import prepare_features
from src.preprocessing import build_preprocessor


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

df = load_data(
    config["data"]["db_path"],
    config["data"]["table_name"]
)

X, y = prepare_features(
    df,
    target_col=config["data"]["target_col"],
    drop_cols=config["data"]["drop_cols"]
)

preprocessor = build_preprocessor(X)

X_transformed = preprocessor.fit_transform(X)

print("Original X shape:", X.shape)
print("Transformed X shape:", X_transformed.shape)