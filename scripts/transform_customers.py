import pandas as pd

# --------------------------------------------------
# 1. Read Bronze data
# --------------------------------------------------

BRONZE_PATH = "data/bronze/customers/customers.parquet"

df = pd.read_parquet(
    BRONZE_PATH,
    dtype_backend="numpy_nullable"
)

print("Bronze customer data loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 2. Customer columns for Silver
# --------------------------------------------------

CUSTOMER_COLUMNS = [
    "customer_id",
    "first_name",
    "last_name",
    "gender",
    "date_of_birth",
    "email",
    "phone",
    "city",
    "state",
    "country",
    "postal_code",
    "registration_date",
    "loyalty_tier",
    "customer_status",
]


# --------------------------------------------------
# 3. Data Quality validations
# --------------------------------------------------

# Customer ID must not be NULL or blank
missing_customer_id = (
    df["customer_id"].isna()
    | df["customer_id"].eq("")
)

print("Missing customer IDs:", missing_customer_id.sum())


# Customer ID must be unique
duplicate_customer_id = df["customer_id"].duplicated(
    keep=False
)

print(
    "Duplicate customer ID rows:",
    duplicate_customer_id.sum()
)


# First name must not be NULL or blank
missing_first_name = (
    df["first_name"].isna()
    | df["first_name"].eq("")
)

print("Missing first names:", missing_first_name.sum())


# Last name must not be NULL or blank
missing_last_name = (
    df["last_name"].isna()
    | df["last_name"].eq("")
)

print("Missing last names:", missing_last_name.sum())


# Gender must be Male, Female or Other
invalid_gender = ~df["gender"].isin(
    ["Male", "Female", "Other"]
)

print("Invalid gender:", invalid_gender.sum())


# Date of birth must be a valid date
dob = pd.to_datetime(
    df["date_of_birth"],
    errors="coerce"
)

invalid_dob = dob.isna()

print("Invalid date of birth:", invalid_dob.sum())


# Email must have valid format
invalid_email = ~df["email"].str.match(
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    na=False
)

print("Invalid email:", invalid_email.sum())


# Phone must be a valid 10-digit Indian mobile number
invalid_phone = ~df["phone"].str.match(
    r"^[6-9]\d{9}$",
    na=False
)

print("Invalid phone:", invalid_phone.sum())


# City must not be NULL or blank
missing_city = (
    df["city"].isna()
    | df["city"].eq("")
)

print("Missing city:", missing_city.sum())


# State must not be NULL or blank
missing_state = (
    df["state"].isna()
    | df["state"].eq("")
)

print("Missing state:", missing_state.sum())


# Country must be India
invalid_country = df["country"].ne("India")

print("Invalid country:", invalid_country.sum())


# Postal code must contain exactly 6 digits
invalid_postal_code = ~df["postal_code"].str.match(
    r"^\d{6}$",
    na=False
)

print(
    "Invalid postal codes:",
    invalid_postal_code.sum()
)


# Registration date must be valid
registration_date = pd.to_datetime(
    df["registration_date"],
    errors="coerce"
)

invalid_registration_date = registration_date.isna()

print(
    "Invalid registration dates:",
    invalid_registration_date.sum()
)


# Loyalty tier must be Bronze, Silver or Gold
invalid_loyalty_tier = ~df["loyalty_tier"].isin(
    ["Bronze", "Silver", "Gold"]
)

print(
    "Invalid loyalty tiers:",
    invalid_loyalty_tier.sum()
)


# Customer status must be Active or Inactive
invalid_status = ~df["customer_status"].isin(
    ["Active", "Inactive"]
)

print(
    "Invalid customer status:",
    invalid_status.sum()
)


# --------------------------------------------------
# 4. Combine all validation rules
# --------------------------------------------------

valid_record = ~(
    missing_customer_id
    | duplicate_customer_id
    | missing_first_name
    | missing_last_name
    | invalid_gender
    | invalid_dob
    | invalid_email
    | invalid_phone
    | missing_city
    | missing_state
    | invalid_country
    | invalid_postal_code
    | invalid_registration_date
    | invalid_loyalty_tier
    | invalid_status
)

print("Valid records:", valid_record.sum())
print("Invalid records:", (~valid_record).sum())

# --------------------------------------------------
# 5. Split valid and invalid records
# --------------------------------------------------

silver_df = df.loc[valid_record, CUSTOMER_COLUMNS]

dq_df = df.loc[~valid_record]

print("Silver records:", len(silver_df))
print("DQ records:", len(dq_df))

# --------------------------------------------------
# 6. Write Silver data
# --------------------------------------------------

SILVER_PATH = "data/silver/customers/customers.parquet"

silver_df.to_parquet(
    SILVER_PATH,
    index=False
)

print("Silver customer data written successfully")



# --------------------------------------------------
# 7. Write DQ / Quarantine data
# --------------------------------------------------

DQ_PATH = "data/dq/customers/customers_dq.parquet"

dq_df.to_parquet(
    DQ_PATH,
    index=False
)

print("DQ customer data written successfully")