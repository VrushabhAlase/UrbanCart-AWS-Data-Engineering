import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BRONZE_PATH = Path("data/bronze/orders/orders.parquet")

SILVER_PATH = Path("data/silver/orders/orders.parquet")
DQ_PATH = Path("data/dq/orders/orders_dq.parquet")


# ============================================================
# LOAD BRONZE DATA
# ============================================================

df = pd.read_parquet(
    BRONZE_PATH,
    dtype_backend="numpy_nullable"
)

print("Bronze order data loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# VALID VALUES
# ============================================================

VALID_ORDER_STATUSES = {
    "Pending",
    "Confirmed",
    "Shipped",
    "Delivered",
    "Cancelled",
    "Returned"
}

VALID_PAYMENT_METHODS = {
    "UPI",
    "Credit Card",
    "Debit Card",
    "Net Banking",
    "Cash on Delivery",
    "Wallet"
}


# ============================================================
# PROCESSING DATE
# ============================================================

PROCESSING_DATE = pd.Timestamp("2026-08-01")


# ============================================================
# BASIC VALIDATIONS
# ============================================================

# 1. Missing Order ID
missing_order_id = (
    df["order_id"].isna()
    | df["order_id"].astype("string").str.strip().eq("")
)

print(
    "Missing order IDs:",
    missing_order_id.sum()
)


# 2. Duplicate Order ID
# Only the duplicate occurrence is considered DQ.
duplicate_order_id = df["order_id"].duplicated(
    keep="first"
)

print(
    "Duplicate order ID rows:",
    duplicate_order_id.sum()
)


# ============================================================
# CUSTOMER ID VALIDATION
# ============================================================

customer_id_pattern = (
    df["customer_id"]
    .astype("string")
    .str.fullmatch(r"C\d{6}")
)

customer_id_number = pd.to_numeric(
    df["customer_id"]
    .astype("string")
    .str[1:],
    errors="coerce"
)

# 3. Missing Customer ID
missing_customer_id = (
    df["customer_id"].isna()
    | df["customer_id"].astype("string").str.strip().eq("")
)

print(
    "Missing customer IDs:",
    missing_customer_id.sum()
)


# 4. Invalid Customer ID
invalid_customer_id = (
    ~customer_id_pattern.fillna(False)
    | customer_id_number.isna()
    | ~customer_id_number.between(1, 50000)
)

print(
    "Invalid customer IDs:",
    invalid_customer_id.sum()
)


# ============================================================
# ORDER DATE VALIDATION
# ============================================================

order_date = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

# 5. Invalid Order Date
invalid_order_date = order_date.isna()

print(
    "Invalid order dates:",
    invalid_order_date.sum()
)


# 6. Future Order Date
future_order_date = (
    order_date.notna()
    & (order_date > PROCESSING_DATE)
)

print(
    "Future order dates:",
    future_order_date.sum()
)


# ============================================================
# ORDER STATUS
# ============================================================

# 7. Invalid Order Status
invalid_order_status = (
    ~df["order_status"].isin(
        VALID_ORDER_STATUSES
    )
)

print(
    "Invalid order statuses:",
    invalid_order_status.sum()
)


# ============================================================
# PAYMENT METHOD
# ============================================================

# 8. Invalid Payment Method
invalid_payment_method = (
    ~df["payment_method"].isin(
        VALID_PAYMENT_METHODS
    )
)

print(
    "Invalid payment methods:",
    invalid_payment_method.sum()
)


# ============================================================
# SHIPPING VALIDATION
# ============================================================

# 9. Missing Shipping City
missing_shipping_city = (
    df["shipping_city"].isna()
    | df["shipping_city"]
    .astype("string")
    .str.strip()
    .eq("")
)

print(
    "Missing shipping cities:",
    missing_shipping_city.sum()
)


# 10. Missing Shipping State
missing_shipping_state = (
    df["shipping_state"].isna()
    | df["shipping_state"]
    .astype("string")
    .str.strip()
    .eq("")
)

print(
    "Missing shipping states:",
    missing_shipping_state.sum()
)


# 11. Invalid Postal Code
postal_code_pattern = (
    df["shipping_postal_code"]
    .astype("string")
    .str.fullmatch(r"\d{6}")
)

invalid_shipping_postal_code = (
    ~postal_code_pattern.fillna(False)
)

print(
    "Invalid shipping postal codes:",
    invalid_shipping_postal_code.sum()
)


# ============================================================
# AMOUNT VALIDATION
# ============================================================

subtotal = pd.to_numeric(
    df["subtotal_amount"],
    errors="coerce"
)

discount = pd.to_numeric(
    df["discount_amount"],
    errors="coerce"
)

shipping_fee = pd.to_numeric(
    df["shipping_fee"],
    errors="coerce"
)

tax = pd.to_numeric(
    df["tax_amount"],
    errors="coerce"
)

total = pd.to_numeric(
    df["total_amount"],
    errors="coerce"
)


# 12. Invalid Subtotal
invalid_subtotal_amount = (
    subtotal.isna()
    | (subtotal <= 0)
)

print(
    "Invalid subtotal amounts:",
    invalid_subtotal_amount.sum()
)


# 13. Invalid Discount
invalid_discount_amount = (
    discount.isna()
    | (discount < 0)
)

print(
    "Invalid discount amounts:",
    invalid_discount_amount.sum()
)


# 14. Discount greater than subtotal
discount_exceeds_subtotal = (
    subtotal.notna()
    & discount.notna()
    & (discount > subtotal)
)

print(
    "Discount exceeds subtotal:",
    discount_exceeds_subtotal.sum()
)


# 15. Invalid Shipping Fee
invalid_shipping_fee = (
    shipping_fee.isna()
    | (shipping_fee < 0)
)

print(
    "Invalid shipping fees:",
    invalid_shipping_fee.sum()
)


# 16. Invalid Tax
invalid_tax_amount = (
    tax.isna()
    | (tax < 0)
)

print(
    "Invalid tax amounts:",
    invalid_tax_amount.sum()
)


# ============================================================
# TOTAL AMOUNT BUSINESS RULE
# ============================================================

expected_total = (
    subtotal
    - discount
    + shipping_fee
    + tax
)

# Round both sides to 2 decimal places
expected_total = expected_total.round(2)
actual_total = total.round(2)

# 17. Invalid Total Amount
invalid_total_amount = (
    total.isna()
    | (total < 0)
    | (
        total.notna()
        & expected_total.notna()
        & (
            actual_total
            != expected_total
        )
    )
)

print(
    "Invalid total amounts:",
    invalid_total_amount.sum()
)


# ============================================================
# CREATE DQ ISSUE
# ============================================================

df["_dq_issue"] = pd.NA


def add_dq_issue(mask, issue):
    """
    Assign the first detected DQ issue
    to each invalid record.
    """

    available = (
        mask
        & df["_dq_issue"].isna()
    )

    df.loc[
        available,
        "_dq_issue"
    ] = issue


# ============================================================
# ASSIGN DQ ISSUES
# ============================================================

add_dq_issue(
    duplicate_order_id,
    "duplicate_order_record"
)

add_dq_issue(
    missing_order_id,
    "missing_required_field_order_id"
)

add_dq_issue(
    missing_customer_id,
    "missing_customer_id"
)

add_dq_issue(
    invalid_customer_id,
    "invalid_customer_id"
)

add_dq_issue(
    invalid_order_date,
    "invalid_order_date"
)

add_dq_issue(
    future_order_date,
    "future_order_date"
)

add_dq_issue(
    invalid_order_status,
    "invalid_order_status"
)

add_dq_issue(
    invalid_payment_method,
    "invalid_payment_method"
)

add_dq_issue(
    missing_shipping_city,
    "missing_shipping_city"
)

add_dq_issue(
    missing_shipping_state,
    "missing_shipping_state"
)

add_dq_issue(
    invalid_shipping_postal_code,
    "invalid_shipping_postal_code"
)

add_dq_issue(
    invalid_subtotal_amount,
    "invalid_subtotal_amount"
)

add_dq_issue(
    invalid_discount_amount,
    "invalid_discount_amount"
)

add_dq_issue(
    discount_exceeds_subtotal,
    "discount_exceeds_subtotal"
)

add_dq_issue(
    invalid_shipping_fee,
    "invalid_shipping_fee"
)

add_dq_issue(
    invalid_tax_amount,
    "invalid_tax_amount"
)

add_dq_issue(
    invalid_total_amount,
    "invalid_total_amount"
)


# ============================================================
# SPLIT SILVER AND DQ
# ============================================================

valid_mask = df["_dq_issue"].isna()

silver_df = df.loc[
    valid_mask
].copy()

dq_df = df.loc[
    ~valid_mask
].copy()


# ============================================================
# REMOVE DQ COLUMN FROM SILVER
# ============================================================

silver_df = silver_df.drop(
    columns=["_dq_issue"]
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

SILVER_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

DQ_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# WRITE SILVER
# ============================================================

silver_df.to_parquet(
    SILVER_PATH,
    index=False
)


# ============================================================
# WRITE DQ
# ============================================================

dq_df.to_parquet(
    DQ_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("========================================")
print("ORDER DATA QUALITY SUMMARY")
print("========================================")

print(
    "Total records:",
    len(df)
)

print(
    "Valid records:",
    len(silver_df)
)

print(
    "Invalid records:",
    len(dq_df)
)

print()
print(
    "Silver rows:",
    len(silver_df)
)

print(
    "Silver columns:",
    len(silver_df.columns)
)

print()
print(
    "DQ rows:",
    len(dq_df)
)

print(
    "DQ columns:",
    len(dq_df.columns)
)

print()
print("Silver columns:")
print(
    silver_df.columns.tolist()
)

print()
print("DQ columns:")
print(
    dq_df.columns.tolist()
)

print()
print("DQ issue distribution:")
print(
    dq_df["_dq_issue"]
    .value_counts(dropna=False)
)

print()
print(
    "_dq_issue present in Silver:",
    "_dq_issue" in silver_df.columns
)

print()
print(
    "Order Silver and DQ data written successfully"
)


# ============================================================
# FINAL SILVER / DQ VALIDATION
# ============================================================

print()
print("========================================")
print("FINAL ORDER DATA VALIDATION")
print("========================================")

silver_check = pd.read_parquet(
    SILVER_PATH,
    dtype_backend="numpy_nullable"
)

dq_check = pd.read_parquet(
    DQ_PATH,
    dtype_backend="numpy_nullable"
)

print("Silver rows:", len(silver_check))
print("Silver columns:", len(silver_check.columns))

print(
    "Silver duplicate order IDs:",
    silver_check["order_id"].duplicated().sum()
)

print()
print("Silver null values:")
print(
    silver_check.isna().sum()
)

print(
    "_dq_issue present in Silver:",
    "_dq_issue" in silver_check.columns
)

print()
print("DQ rows:", len(dq_check))

print(
    "_dq_issue present in DQ:",
    "_dq_issue" in dq_check.columns
)

print(
    "DQ issue null values:",
    dq_check["_dq_issue"].isna().sum()
)


# ============================================================
# FINAL CUSTOMER ID VALIDATION
# ============================================================

silver_customer_pattern = (
    silver_check["customer_id"]
    .astype("string")
    .str.fullmatch(r"C\d{6}")
)

silver_customer_number = pd.to_numeric(
    silver_check["customer_id"]
    .astype("string")
    .str[1:],
    errors="coerce"
)

invalid_silver_customers = (
    ~silver_customer_pattern.fillna(False)
    | silver_customer_number.isna()
    | ~silver_customer_number.between(1, 50000)
)

print(
    "Invalid customer IDs in Silver:",
    invalid_silver_customers.sum()
)


# ============================================================
# FINAL TOTAL AMOUNT VALIDATION
# ============================================================

silver_subtotal = pd.to_numeric(
    silver_check["subtotal_amount"],
    errors="coerce"
)

silver_discount = pd.to_numeric(
    silver_check["discount_amount"],
    errors="coerce"
)

silver_shipping = pd.to_numeric(
    silver_check["shipping_fee"],
    errors="coerce"
)

silver_tax = pd.to_numeric(
    silver_check["tax_amount"],
    errors="coerce"
)

silver_total = pd.to_numeric(
    silver_check["total_amount"],
    errors="coerce"
)

expected_silver_total = (
    silver_subtotal
    - silver_discount
    + silver_shipping
    + silver_tax
).round(2)

invalid_silver_totals = (
    silver_total.round(2)
    != expected_silver_total
)

print(
    "Invalid total amounts in Silver:",
    invalid_silver_totals.sum()
)