import pandas as pd

# ===== 1. LOAD DATA =====
csv_path = "data.csv"   # ปรับ path ถ้าอยู่โฟลเดอร์อื่น
df = pd.read_csv(csv_path, sep=";")

print("Dataset shape:", df.shape)

# ===== 2. BASIC COLUMN SUMMARY =====
summary = pd.DataFrame({
    "column_name": df.columns,
    "dtype": df.dtypes.astype(str),
    "non_null_count": df.notnull().sum().values,
    "null_count": df.isnull().sum().values,
    "null_ratio": (df.isnull().mean() * 100).round(2),
    "unique_values": df.nunique().values
})

# ===== 3. SAMPLE VALUES (ช่วยดูว่าข้อมูลหน้าตาเป็นยังไง) =====
sample_values = {}
for col in df.columns:
    sample_values[col] = df[col].dropna().unique()[:5]

summary["sample_values"] = summary["column_name"].map(sample_values)

# ===== 4. SPLIT NUMERIC / CATEGORICAL =====
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_summary = summary[summary["column_name"].isin(numeric_cols)]
categorical_summary = summary[summary["column_name"].isin(categorical_cols)]

# ===== 5. EXPORT TO EXCEL =====
output_excel = "data_column_overview.xlsx"

with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
    summary.to_excel(writer, sheet_name="All_Columns", index=False)
    numeric_summary.to_excel(writer, sheet_name="Numeric_Columns", index=False)
    categorical_summary.to_excel(writer, sheet_name="Categorical_Columns", index=False)

print(f"Exported column overview to: {output_excel}")
