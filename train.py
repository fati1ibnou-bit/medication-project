import os
import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# 1 CONNEXION MYSQL

user = "root"
password = "your_password"  # ou os.getenv("DB_PASSWORD")
host = "localhost"
database = "pharmacy_db"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

# 2 CHARGER DATASET

path = "data/ml_dataset.csv"

if not os.path.exists(path):
    raise FileNotFoundError("Dataset introuvable. Lance export_data.py")

df = pd.read_csv(path)

print(f"Dataset chargé : {df.shape}")

# 3 NETTOYAGE

df = df.drop_duplicates()
df = df.dropna()

# Conversion date 
if "sale_date" in df.columns:
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["day_of_week"] = df["sale_date"].dt.dayofweek
    df["month"] = df["sale_date"].dt.month


# 4 résumer des ventes par jour et par médicament

df["date_day"] = df["sale_date"].dt.date

df_daily = df.groupby(["medication_id", "date_day"])["quantity_sold"].sum().reset_index()
df_daily["date_day"] = pd.to_datetime(df_daily["date_day"])


# 5. remplir les jours manquants 

all_dates = pd.date_range(df_daily["date_day"].min(), df_daily["date_day"].max(), freq="D")
all_meds = df_daily["medication_id"].unique()

index = pd.MultiIndex.from_product([all_meds, all_dates], names=["medication_id", "date_day"])
df_full = pd.DataFrame(index=index).reset_index()

df_daily = pd.merge(df_full, df_daily, on=["medication_id", "date_day"], how="left")
df_daily["quantity_sold"] = df_daily["quantity_sold"].fillna(0)


# 6. FEATURE ENGINEERING 

df_daily = df_daily.sort_values(["medication_id", "date_day"])

# rolling sécurisé (important correction)
df_daily["rolling_7"] = df_daily.groupby("medication_id")["quantity_sold"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

df_daily["rolling_30"] = df_daily.groupby("medication_id")["quantity_sold"].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
)

df_daily["day_of_week"] = df_daily["date_day"].dt.dayofweek
df_daily["month"] = df_daily["date_day"].dt.month

# TARGET (demande J+1)
df_daily["target"] = df_daily.groupby("medication_id")["quantity_sold"].shift(-1)

# suppression NaN
df_daily = df_daily.dropna()

print(f"Dataset final ML : {df_daily.shape}")


# 7. FEATURES / TARGET

features = ["rolling_7", "rolling_30", "day_of_week", "month"]

X = df_daily[features]
y = df_daily["target"]

# 8. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. MODÈLE ML

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

print(" Training model...")
model.fit(X_train, y_train)

# 10. ÉVALUATION

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n MODEL EVALUATION")
print(f"MAE  : {round(mae, 2)}")
print(f"RMSE : {round(rmse, 2)}")

# 11 SAUVEGARDE MODELE

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("\n Model saved in model/model.pkl")


# 12. RECOMMANDATION STOCK

print("\n Stock recommendation generating...")

query = """
SELECT 
    m.id AS medication_id,
    m.name,
    m.min_stock_level,
    sm.current_stock
FROM medications m
JOIN stock_movements sm ON m.id = sm.medication_id
"""

df_stock = pd.read_sql(query, engine)

latest = df_daily.sort_values("date_day").groupby("medication_id").tail(1)

df_rec = pd.merge(df_stock, latest, on="medication_id", how="inner")

df_rec["predicted_demand"] = model.predict(df_rec[features])

def status(row):
    gap = row["current_stock"] - row["predicted_demand"]

    if gap <= row["min_stock_level"]:
        return "URGENT RESTOCK"
    elif gap <= row["min_stock_level"] * 1.3:
        return "WARNING"
    else:
        return "OK"

df_rec["status"] = df_rec.apply(status, axis=1)

os.makedirs("output", exist_ok=True)

df_rec.to_csv("output/restock.csv", index=False)

print("Recommendations saved in output/restock.csv")
print("Ready for API step")