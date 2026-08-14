import pandas as pd

# --------------------------------------------------
# 1. Read Bronze Sellers data
# --------------------------------------------------

BRONZE_PATH = "data/bronze/sellers/sellers.parquet"

df = pd.read_parquet(
    BRONZE_PATH,
    dtype_backend="numpy_nullable"
)

print("Bronze seller data loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 2. Seller columns for Silver
# --------------------------------------------------

SELLER_COLUMNS = [
    "seller_id",
    "seller_name",
    "email",
    "phone",
    "city",
    "state",
    "country",
    "seller_rating",
    "seller_status",
    "joined_date",
]


# --------------------------------------------------
# 3. Data Quality validations
# --------------------------------------------------

# Seller ID must not be NULL or blank
missing_seller_id = (
    df["seller_id"].isna()
    | df["seller_id"].eq("")
)

print("Missing seller IDs:", missing_seller_id.sum())


# Seller ID must be unique
duplicate_seller_id = df["seller_id"].duplicated(
    keep=False
)

print(
    "Duplicate seller ID rows:",
    duplicate_seller_id.sum()
)


# Seller name must not be NULL or blank
missing_seller_name = (
    df["seller_name"].isna()
    | df["seller_name"].eq("")
)

print(
    "Missing seller names:",
    missing_seller_name.sum()
)


# Email must have valid format
invalid_email = ~df["email"].str.match(
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    na=False
)

print(
    "Invalid emails:",
    invalid_email.sum()
)


# Phone must be a valid 10-digit Indian mobile number
invalid_phone = ~df["phone"].str.match(
    r"^[6-9]\d{9}$",
    na=False
)

print(
    "Invalid phones:",
    invalid_phone.sum()
)


# City must not be NULL or blank
missing_city = (
    df["city"].isna()
    | df["city"].eq("")
)

print(
    "Missing cities:",
    missing_city.sum()
)


# State must not be NULL or blank
missing_state = (
    df["state"].isna()
    | df["state"].eq("")
)

print(
    "Missing states:",
    missing_state.sum()
)


# Country must be India
invalid_country = df["country"].ne("India")

print(
    "Invalid countries:",
    invalid_country.sum()
)


# Seller rating must be between 1 and 5
invalid_rating = (
    df["seller_rating"].isna()
    | (df["seller_rating"] < 1)
    | (df["seller_rating"] > 5)
)

print(
    "Invalid seller ratings:",
    invalid_rating.sum()
)


# Seller status must be Active or Inactive
invalid_status = ~df["seller_status"].isin(
    ["Active", "Inactive"]
)

print(
    "Invalid seller statuses:",
    invalid_status.sum()
)


# Joined date must be valid
joined_date = pd.to_datetime(
    df["joined_date"],
    errors="coerce"
)

invalid_joined_date = joined_date.isna()

print(
    "Invalid joined dates:",
    invalid_joined_date.sum()
)


# Joined date cannot be in the future
processing_date = pd.Timestamp("2026-08-01")

future_joined_date = (
    joined_date > processing_date
)

print(
    "Future joined dates:",
    future_joined_date.sum()
)


# --------------------------------------------------
# 4. Combine all validation rules
# --------------------------------------------------

valid_record = ~(
    missing_seller_id
    | duplicate_seller_id
    | missing_seller_name
    | invalid_email
    | invalid_phone
    | missing_city
    | missing_state
    | invalid_country
    | invalid_rating
    | invalid_status
    | invalid_joined_date
    | future_joined_date
)

print("Valid records:", valid_record.sum())
print("Invalid records:", (~valid_record).sum())

# --------------------------------------------------
# 5. Split valid and invalid records
# --------------------------------------------------

silver_df = df.loc[valid_record, SELLER_COLUMNS]

dq_df = df.loc[~valid_record]

print("Silver records:", len(silver_df))
print("DQ records:", len(dq_df))

# --------------------------------------------------
# 6. Write Silver data
# --------------------------------------------------

SILVER_PATH = "data/silver/sellers/sellers.parquet"

silver_df.to_parquet(
    SILVER_PATH,
    index=False
)

print("Silver seller data written successfully")


# --------------------------------------------------
# 7. Write DQ / Quarantine data
# --------------------------------------------------

DQ_PATH = "data/dq/sellers/sellers_dq.parquet"

dq_df.to_parquet(
    DQ_PATH,
    index=False
)

print("DQ seller data written successfully")