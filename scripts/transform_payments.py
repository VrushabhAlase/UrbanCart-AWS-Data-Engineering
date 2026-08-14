import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

RAW_PATH = Path("data/raw/payments.parquet")

ORDERS_PATH = Path(
    "data/silver/orders/orders.parquet"
)

BRONZE_PATH = Path(
    "data/bronze/Payments/payments.parquet"
)

SILVER_PATH = Path(
    "data/silver/Payments/payments.parquet"
)

DQ_PATH = Path(
    "data/dq/Payments/payments_dq.parquet"
)


# ============================================================
# LOAD RAW PAYMENT DATA
# ============================================================

df = pd.read_parquet(
    RAW_PATH,
    dtype_backend="numpy_nullable"
)

print("Raw payment data loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Column names:", df.columns.tolist())


# ============================================================
# LOAD ORDERS SILVER
# ============================================================

orders = pd.read_parquet(
    ORDERS_PATH,
    dtype_backend="numpy_nullable"
)

print()
print("Orders Silver data loaded successfully")
print("Orders rows:", len(orders))


# ============================================================
# CREATE BRONZE DIRECTORY
# ============================================================

BRONZE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# WRITE BRONZE DATA
# ============================================================

df.to_parquet(
    BRONZE_PATH,
    index=False
)

print("Bronze payment data written successfully")


# ============================================================
# VALID VALUES
# ============================================================

VALID_PAYMENT_METHODS = {
    "UPI",
    "Credit Card",
    "Debit Card",
    "Net Banking",
    "Cash on Delivery",
    "Wallet"
}

VALID_PAYMENT_STATUSES = {
    "Pending",
    "Success",
    "Failed",
    "Refunded",
    "Cancelled"
}

VALID_CURRENCIES = {
    "INR"
}

VALID_GATEWAYS = {
    "Razorpay",
    "PayU",
    "Cashfree",
    "Stripe"
}

PROCESSING_DATE = pd.Timestamp(
    "2026-08-01"
)


# ============================================================
# PAYMENT ID VALIDATION
# ============================================================

missing_payment_id = (
    df["payment_id"].isna()
    | df["payment_id"]
    .astype("string")
    .str.strip()
    .eq("")
)

duplicate_payment_id = (
    df["payment_id"].duplicated(
        keep="first"
    )
)

print()
print(
    "Missing payment IDs:",
    missing_payment_id.sum()
)

print(
    "Duplicate payment ID rows:",
    duplicate_payment_id.sum()
)


# ============================================================
# ORDER ID VALIDATION
# ============================================================

missing_order_id = (
    df["order_id"].isna()
    | df["order_id"]
    .astype("string")
    .str.strip()
    .eq("")
)

# Use actual Orders Silver IDs for referential validation.
valid_order_ids = set(
    orders["order_id"]
    .dropna()
    .astype("string")
)

invalid_order_id = (
    ~df["order_id"]
    .astype("string")
    .isin(valid_order_ids)
)

print(
    "Missing order IDs:",
    missing_order_id.sum()
)

print(
    "Invalid order IDs:",
    invalid_order_id.sum()
)


# ============================================================
# PREPARE ORDER REFERENCE DATA
# ============================================================

order_reference = orders[
    [
        "order_id",
        "order_date",
        "total_amount"
    ]
].copy()

order_reference["order_date"] = pd.to_datetime(
    order_reference["order_date"],
    errors="coerce"
)

order_reference["total_amount"] = pd.to_numeric(
    order_reference["total_amount"],
    errors="coerce"
)

payment_with_orders = df.merge(
    order_reference,
    on="order_id",
    how="left",
    suffixes=(
        "",
        "_order"
    )
)


# ============================================================
# PAYMENT DATE VALIDATION
# ============================================================

payment_date = pd.to_datetime(
    df["payment_date"],
    errors="coerce"
)

invalid_payment_date = (
    payment_date.isna()
)

print(
    "Invalid payment dates:",
    invalid_payment_date.sum()
)


# ============================================================
# FUTURE PAYMENT DATE
# ============================================================

future_payment_date = (
    payment_date.notna()
    & (
        payment_date
        > PROCESSING_DATE
    )
)

print(
    "Future payment dates:",
    future_payment_date.sum()
)


# ============================================================
# PAYMENT BEFORE ORDER DATE
# ============================================================

related_order_date = pd.to_datetime(
    payment_with_orders["order_date"],
    errors="coerce"
)

payment_before_order_date = (
    payment_date.notna()
    & related_order_date.notna()
    & (
        payment_date
        < related_order_date
    )
)

print(
    "Payment before order date:",
    payment_before_order_date.sum()
)


# ============================================================
# PAYMENT METHOD
# ============================================================

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
# PAYMENT STATUS
# ============================================================

invalid_payment_status = (
    ~df["payment_status"].isin(
        VALID_PAYMENT_STATUSES
    )
)

print(
    "Invalid payment statuses:",
    invalid_payment_status.sum()
)


# ============================================================
# TRANSACTION REFERENCE
# ============================================================

transaction_reference = (
    df["transaction_reference"]
    .astype("string")
    .str.strip()
)

# NULL is allowed for Pending, Failed and Cancelled.
#
# If a transaction reference is provided, it must follow
# the expected transaction-reference format.

invalid_transaction_reference = (
    transaction_reference.notna()
    & transaction_reference.ne("")
    & ~transaction_reference.str.match(
        r"^TXN\d+$",
        na=False
    )
)

print(
    "Invalid transaction references:",
    invalid_transaction_reference.sum()
)


# ============================================================
# PAYMENT AMOUNT
# ============================================================

amount = pd.to_numeric(
    df["amount"],
    errors="coerce"
)

invalid_amount = (
    amount.isna()
    | (amount <= 0)
)

print(
    "Invalid payment amounts:",
    invalid_amount.sum()
)


# ============================================================
# CURRENCY
# ============================================================

invalid_currency = (
    ~df["currency"].isin(
        VALID_CURRENCIES
    )
)

print(
    "Invalid currencies:",
    invalid_currency.sum()
)


# ============================================================
# PAYMENT GATEWAY
# ============================================================

invalid_payment_gateway = (
    ~df["payment_gateway"].isin(
        VALID_GATEWAYS
    )
)

print(
    "Invalid payment gateways:",
    invalid_payment_gateway.sum()
)


# ============================================================
# REQUIRED TRANSACTION REFERENCE
# ============================================================

required_transaction_reference = (
    df["payment_status"]
    .isin(
        {
            "Success",
            "Refunded"
        }
    )
)

missing_transaction_reference = (
    required_transaction_reference
    & (
        transaction_reference.isna()
        | transaction_reference.eq("")
    )
)

print(
    "Missing transaction references:",
    missing_transaction_reference.sum()
)


# ============================================================
# PAYMENT AMOUNT RECONCILIATION
# ============================================================

related_order_total = pd.to_numeric(
    payment_with_orders[
        "total_amount"
    ],
    errors="coerce"
)

amount_mismatch_with_order = (
    amount.notna()
    & related_order_total.notna()
    & (
        amount.round(2)
        != related_order_total.round(2)
    )
)

print(
    "Payment amount mismatches with Orders:",
    amount_mismatch_with_order.sum()
)


# ============================================================
# CREATE DQ ISSUE COLUMN
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
    duplicate_payment_id,
    "duplicate_payment_record"
)

add_dq_issue(
    missing_payment_id,
    "missing_payment_id"
)

add_dq_issue(
    missing_order_id,
    "missing_order_id"
)

add_dq_issue(
    invalid_order_id,
    "invalid_order_id"
)

add_dq_issue(
    invalid_payment_date,
    "invalid_payment_date"
)

add_dq_issue(
    future_payment_date,
    "future_payment_date"
)

add_dq_issue(
    payment_before_order_date,
    "payment_before_order_date"
)

add_dq_issue(
    invalid_payment_method,
    "invalid_payment_method"
)

add_dq_issue(
    invalid_payment_status,
    "invalid_payment_status"
)

add_dq_issue(
    invalid_transaction_reference,
    "invalid_transaction_reference"
)

add_dq_issue(
    invalid_amount,
    "invalid_amount"
)

add_dq_issue(
    invalid_currency,
    "invalid_currency"
)

add_dq_issue(
    invalid_payment_gateway,
    "invalid_payment_gateway"
)

add_dq_issue(
    missing_transaction_reference,
    "missing_transaction_reference"
)

add_dq_issue(
    amount_mismatch_with_order,
    "amount_mismatch_with_order"
)


# ============================================================
# SPLIT SILVER AND DQ
# ============================================================

valid_mask = (
    df["_dq_issue"].isna()
)

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
# DATA QUALITY SUMMARY
# ============================================================

print()
print("========================================")
print("PAYMENT DATA QUALITY SUMMARY")
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
    .value_counts(
        dropna=False
    )
)

print()
print(
    "_dq_issue present in Silver:",
    "_dq_issue" in silver_df.columns
)

print()
print(
    "Payment Silver and DQ data written successfully"
)


# ============================================================
# FINAL SILVER / DQ VALIDATION
# ============================================================

print()
print("========================================")
print("FINAL PAYMENT DATA VALIDATION")
print("========================================")

silver_check = pd.read_parquet(
    SILVER_PATH,
    dtype_backend="numpy_nullable"
)

dq_check = pd.read_parquet(
    DQ_PATH,
    dtype_backend="numpy_nullable"
)

print(
    "Silver rows:",
    len(silver_check)
)

print(
    "Silver columns:",
    len(silver_check.columns)
)

print(
    "Silver duplicate payment IDs:",
    silver_check[
        "payment_id"
    ].duplicated().sum()
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
print(
    "DQ rows:",
    len(dq_check)
)

print(
    "_dq_issue present in DQ:",
    "_dq_issue" in dq_check.columns
)

print(
    "DQ issue null values:",
    dq_check[
        "_dq_issue"
    ].isna().sum()
)


# ============================================================
# FINAL ORDER REFERENTIAL VALIDATION
# ============================================================

silver_invalid_orders = (
    ~silver_check[
        "order_id"
    ]
    .astype("string")
    .isin(valid_order_ids)
)

print(
    "Invalid order IDs in Silver:",
    silver_invalid_orders.sum()
)


# ============================================================
# FINAL PAYMENT DATE VALIDATION
# ============================================================

silver_payment_dates = pd.to_datetime(
    silver_check[
        "payment_date"
    ],
    errors="coerce"
)

print(
    "Invalid payment dates in Silver:",
    silver_payment_dates.isna().sum()
)

print(
    "Future payment dates in Silver:",
    (
        silver_payment_dates
        > PROCESSING_DATE
    ).sum()
)


# ============================================================
# FINAL PAYMENT AMOUNT RECONCILIATION
# ============================================================

final_payment_orders = silver_check.merge(
    orders[
        [
            "order_id",
            "total_amount"
        ]
    ],
    on="order_id",
    how="left",
    suffixes=(
        "_payment",
        "_order"
    )
)

final_payment_amount = pd.to_numeric(
    final_payment_orders[
        "amount"
    ],
    errors="coerce"
)

final_order_amount = pd.to_numeric(
    final_payment_orders[
        "total_amount"
    ],
    errors="coerce"
)

invalid_silver_amounts = (
    final_payment_amount.round(2)
    != final_order_amount.round(2)
)

print(
    "Payment amount mismatches in Silver:",
    invalid_silver_amounts.sum()
)


# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print()
print(
    "Final Payment validation completed successfully"
)

