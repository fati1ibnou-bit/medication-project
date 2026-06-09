import os
import pandas as pd
from sqlalchemy import create_engine


# 1 CONNEXION MYSQL

user = "root"
password = ""
host = "localhost"
database = "pharmacy_db"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

# 2 CHARGER TABLE VENTES

query = """
SELECT 
    s.id AS sale_id,
    s.sale_date,
    s.medication_id,
    m.name AS medication_name,
    m.category AS medication_category,
    s.quantity_sold,
    s.total_price
FROM sales s
JOIN medications m ON s.medication_id = m.id
"""

df = pd.read_sql(query, engine)


# 3. CRÉER DOSSIER data

os.makedirs("data", exist_ok=True)


# 4 EXPORT CSV

df.to_csv("data/ml_dataset.csv", index=False)

print(f"✅ Export terminé avec succès ! ({len(df)} lignes exportées dans data/ml_dataset.csv)")

print(df.head())
print(df.shape)