from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    trim,
    when,
    expr,
    row_number,
)
from pyspark.sql.window import Window


# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("UrbanCart-Payments-Silver")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/home/vrual/UrbanCart-AWS-Data-Engineering")

RAW_PATH = PROJECT_ROOT / "data/raw/payments.parquet"
ORDERS_PATH = PROJECT_ROOT / "data/silver/orders/orders.parquet"
BRONZE_PATH = PROJECT_ROOT / "data/bronze/Payments/payments.parquet"
SILVER_PATH = PROJECT_ROOT / "data/silver/Payments/payments.parquet"
DQ_PATH = PROJECT_ROOT / "data/dq/Payments/payments_dq.parquet"

PROCESSING_DATE = "2026-08-01"


# ============================================================
# VALID VALUES
# ============================================================

VALID_PAYMENT_METHODS = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Net Banking",
    "Cash on Delivery",
    "Wallet",
]

VALID_PAYMENT_STATUSES = [
    "Pending",
    "Success",
    "Failed",
    "Refunded",
    "Cancelled",
]

VALID_CURRENCIES = ["INR"]

VALID_GATEWAYS = [
    "Razorpay",
    "PayU",
    "Cashfree",
    "Stripe",
]


# ============================================================
# LOAD RAW PAYMENT DATA
# ============================================================

df = spark.read.parquet(str(RAW_PATH))

print()
print("========================================")
print("RAW PAYMENT DATA")
print("========================================")
print("Rows:", df.count())
print("Columns:", len(df.columns))
print("Column names:", df.columns)


# Source _dq_issue is not part of the payment Silver business
# validation. It will be recreated by this Spark job.
if "_dq_issue" in df.columns:
    df = df.drop("_dq_issue")


# ============================================================
# LOAD ORDERS SILVER
# ============================================================

orders = spark.read.parquet(str(ORDERS_PATH))

print()
print("Orders Silver data loaded successfully")
print("Orders rows:", orders.count())


# ============================================================
# WRITE BRONZE
# ============================================================

BRONZE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.write.mode("overwrite").parquet(
    str(BRONZE_PATH)
)

print("Bronze payment data written successfully")


# Reload exactly what was written to Bronze.
df = spark.read.parquet(str(BRONZE_PATH))


# ============================================================
# BRONZE DATA INFORMATION
# ============================================================

print()
print("========================================")
print("BRONZE PAYMENT DATA")
print("========================================")
print("Rows:", df.count())
print("Columns:", len(df.columns))
print("Column names:", df.columns)

print()
print("Schema:")
df.printSchema()

print()
print("Sample records:")
df.show(10, truncate=False)


# ============================================================
# PREPARE NORMALIZED EXPRESSIONS
# ============================================================

payment_id_string = trim(
    col("payment_id").cast("string")
)

order_id_string = trim(
    col("order_id").cast("string")
)

transaction_reference_string = trim(
    col("transaction_reference").cast("string")
)


# ============================================================
# PAYMENT ID VALIDATION
# ============================================================

missing_payment_id = (
    col("payment_id").isNull()
    | (payment_id_string == "")
)

# IMPORTANT:
# The old Pandas implementation uses duplicated(keep="first").
# Therefore the FIRST occurrence of a payment_id is retained,
# while subsequent occurrences are marked as duplicates.
payment_id_window = (
    Window
    .partitionBy("payment_id")
    .orderBy(lit(1))
)

df = df.withColumn(
    "_payment_id_row_number",
    row_number().over(payment_id_window)
)

duplicate_payment_id = (
    col("payment_id").isNotNull()
    & (col("_payment_id_row_number") > 1)
)

print()
print("Missing payment IDs:", df.filter(missing_payment_id).count())
print(
    "Duplicate payment ID rows:",
    df.filter(duplicate_payment_id).count()
)


# ============================================================
# ORDER ID VALIDATION
# ============================================================

missing_order_id = (
    col("order_id").isNull()
    | (order_id_string == "")
)

# Actual Orders Silver IDs are used for referential validation.
valid_order_ids = (
    orders
    .select(
        trim(
            col("order_id").cast("string")
        ).alias("_valid_order_id")
    )
    .where(col("_valid_order_id").isNotNull())
    .distinct()
)

df = df.join(
    valid_order_ids.withColumn(
        "_order_exists",
        lit(1),
    ),
    order_id_string == col("_valid_order_id"),
    how="left",
)

invalid_order_id = (
    ~missing_order_id
    & col("_order_exists").isNull()
)

print(
    "Missing order IDs:",
    df.filter(missing_order_id).count()
)

print(
    "Invalid order IDs:",
    df.filter(invalid_order_id).count()
)


# ============================================================
# ORDER REFERENCE DATA
# ============================================================

# Use try_to_timestamp so malformed order dates cannot crash
# the entire Spark job under ANSI mode.
order_reference = (
    orders
    .select(
        trim(
            col("order_id").cast("string")
        ).alias("_reference_order_id"),

        expr(
            "try_to_timestamp(cast(order_date as string))"
        ).alias("_reference_order_date"),

        expr(
            "try_cast(total_amount AS DOUBLE)"
        ).alias("_reference_order_total"),
    )
)

df = df.join(
    order_reference,
    order_id_string == col("_reference_order_id"),
    how="left",
)


# ============================================================
# PAYMENT DATE VALIDATION
# ============================================================

payment_date = expr(
    "try_to_timestamp(cast(payment_date as string))"
)

processing_timestamp = expr(
    f"try_to_timestamp('{PROCESSING_DATE}')"
)

invalid_payment_date = (
    payment_date.isNull()
)

future_payment_date = (
    payment_date.isNotNull()
    & (payment_date > processing_timestamp)
)

print(
    "Invalid payment dates:",
    df.filter(invalid_payment_date).count()
)

print(
    "Future payment dates:",
    df.filter(future_payment_date).count()
)


# ============================================================
# PAYMENT BEFORE ORDER DATE
# ============================================================

payment_before_order_date = (
    payment_date.isNotNull()
    & col("_reference_order_date").isNotNull()
    & (
        payment_date
        < col("_reference_order_date")
    )
)

print(
    "Payment before order date:",
    df.filter(payment_before_order_date).count()
)


# ============================================================
# PAYMENT METHOD
# ============================================================

invalid_payment_method = (
    col("payment_method").isNull()
    | ~col("payment_method").isin(
        VALID_PAYMENT_METHODS
    )
)

print(
    "Invalid payment methods:",
    df.filter(invalid_payment_method).count()
)


# ============================================================
# PAYMENT STATUS
# ============================================================

invalid_payment_status = (
    col("payment_status").isNull()
    | ~col("payment_status").isin(
        VALID_PAYMENT_STATUSES
    )
)

print(
    "Invalid payment statuses:",
    df.filter(invalid_payment_status).count()
)


# ============================================================
# TRANSACTION REFERENCE
# ============================================================

# NULL/blank transaction reference is allowed for
# Pending, Failed and Cancelled.
#
# If a transaction reference is supplied, it must be
# TXN followed by one or more digits.

invalid_transaction_reference = (
    transaction_reference_string.isNotNull()
    & (transaction_reference_string != "")
    & ~transaction_reference_string.rlike(
        r"^TXN\d+$"
    )
)

print(
    "Invalid transaction references:",
    df.filter(
        invalid_transaction_reference
    ).count()
)


# ============================================================
# PAYMENT AMOUNT
# ============================================================

amount_number = expr(
    "try_cast(amount AS DOUBLE)"
)

invalid_amount = (
    amount_number.isNull()
    | (amount_number <= 0)
)

print(
    "Invalid payment amounts:",
    df.filter(invalid_amount).count()
)


# ============================================================
# CURRENCY
# ============================================================

invalid_currency = (
    col("currency").isNull()
    | ~col("currency").isin(
        VALID_CURRENCIES
    )
)

print(
    "Invalid currencies:",
    df.filter(invalid_currency).count()
)


# ============================================================
# PAYMENT GATEWAY
# ============================================================

invalid_payment_gateway = (
    col("payment_gateway").isNull()
    | ~col("payment_gateway").isin(
        VALID_GATEWAYS
    )
)

print(
    "Invalid payment gateways:",
    df.filter(
        invalid_payment_gateway
    ).count()
)


# ============================================================
# REQUIRED TRANSACTION REFERENCE
# ============================================================

required_transaction_reference = (
    col("payment_status").isin(
        ["Success", "Refunded"]
    )
)

missing_transaction_reference = (
    required_transaction_reference
    & (
        transaction_reference_string.isNull()
        | (transaction_reference_string == "")
    )
)

print(
    "Missing transaction references:",
    df.filter(
        missing_transaction_reference
    ).count()
)


# ============================================================
# PAYMENT AMOUNT RECONCILIATION
# ============================================================

amount_mismatch_with_order = (
    amount_number.isNotNull()
    & col("_reference_order_total").isNotNull()
    & (
        expr(
            "round(try_cast(amount AS DOUBLE), 2)"
        )
        != expr(
            "round(_reference_order_total, 2)"
        )
    )
)

print(
    "Payment amount mismatches with Orders:",
    df.filter(
        amount_mismatch_with_order
    ).count()
)


# ============================================================
# CREATE DQ ISSUE
# ============================================================

# The order here intentionally matches the old Pandas script.
# First matching rule wins.

df = df.withColumn(
    "_dq_issue",
    when(
        duplicate_payment_id,
        lit("duplicate_payment_record"),
    )
    .when(
        missing_payment_id,
        lit("missing_payment_id"),
    )
    .when(
        missing_order_id,
        lit("missing_order_id"),
    )
    .when(
        invalid_order_id,
        lit("invalid_order_id"),
    )
    .when(
        invalid_payment_date,
        lit("invalid_payment_date"),
    )
    .when(
        future_payment_date,
        lit("future_payment_date"),
    )
    .when(
        payment_before_order_date,
        lit("payment_before_order_date"),
    )
    .when(
        invalid_payment_method,
        lit("invalid_payment_method"),
    )
    .when(
        invalid_payment_status,
        lit("invalid_payment_status"),
    )
    .when(
        invalid_transaction_reference,
        lit("invalid_transaction_reference"),
    )
    .when(
        invalid_amount,
        lit("invalid_amount"),
    )
    .when(
        invalid_currency,
        lit("invalid_currency"),
    )
    .when(
        invalid_payment_gateway,
        lit("invalid_payment_gateway"),
    )
    .when(
        missing_transaction_reference,
        lit("missing_transaction_reference"),
    )
    .when(
        amount_mismatch_with_order,
        lit("amount_mismatch_with_order"),
    )
)


# ============================================================
# SPLIT SILVER / DQ
# ============================================================

silver_df = (
    df
    .filter(col("_dq_issue").isNull())
    .select(
        "payment_id",
        "order_id",
        "payment_date",
        "payment_method",
        "payment_status",
        "transaction_reference",
        "amount",
        "currency",
        "payment_gateway",
    )
)

dq_df = (
    df
    .filter(col("_dq_issue").isNotNull())
    .drop(
        "_payment_id_row_number",
        "_valid_order_id",
        "_order_exists",
        "_reference_order_id",
        "_reference_order_date",
        "_reference_order_total",
    )
)


# ============================================================
# COUNTS
# ============================================================

total_records = df.count()
silver_records = silver_df.count()
dq_records = dq_df.count()


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

print()
print("========================================")
print("PAYMENT DATA QUALITY SUMMARY")
print("========================================")

print("Total records:", total_records)
print("Valid records:", silver_records)
print("Invalid records:", dq_records)

print()
print("DQ issue distribution:")

(
    dq_df
    .groupBy("_dq_issue")
    .count()
    .orderBy(col("count").desc())
    .show(truncate=False)
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

SILVER_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DQ_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# WRITE SILVER
# ============================================================

silver_df.write.mode("overwrite").parquet(
    str(SILVER_PATH)
)


# ============================================================
# WRITE DQ
# ============================================================

dq_df.write.mode("overwrite").parquet(
    str(DQ_PATH)
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

print()
print("========================================")
print("PAYMENT SILVER / DQ OUTPUT")
print("========================================")

print("Silver rows:", silver_records)
print("Silver columns:", len(silver_df.columns))

print("DQ rows:", dq_records)
print("DQ columns:", len(dq_df.columns))

print()
print("Silver columns:")
print(silver_df.columns)

print()
print("DQ columns:")
print(dq_df.columns)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("========================================")
print("FINAL PAYMENT DATA VALIDATION")
print("========================================")

silver_check = spark.read.parquet(
    str(SILVER_PATH)
)

dq_check = spark.read.parquet(
    str(DQ_PATH)
)


# ------------------------------------------------------------
# SILVER DUPLICATES
# ------------------------------------------------------------

silver_duplicate_payment_ids = (
    silver_check
    .groupBy("payment_id")
    .count()
    .filter(col("count") > 1)
    .count()
)

print(
    "Silver duplicate payment IDs:",
    silver_duplicate_payment_ids
)


# ------------------------------------------------------------
# SILVER NULL VALUES
# ------------------------------------------------------------

print()
print("Silver null values:")

silver_check.select(
    [
        expr(
            f"sum(CASE WHEN `{column_name}` IS NULL THEN 1 ELSE 0 END)"
        ).alias(column_name)
        for column_name in silver_check.columns
    ]
).show()


# ------------------------------------------------------------
# DQ COLUMN CHECK
# ------------------------------------------------------------

print(
    "_dq_issue present in Silver:",
    "_dq_issue" in silver_check.columns
)

print(
    "_dq_issue present in DQ:",
    "_dq_issue" in dq_check.columns
)

print(
    "DQ issue null values:",
    dq_check
    .filter(col("_dq_issue").isNull())
    .count()
)


# ------------------------------------------------------------
# FINAL ORDER REFERENTIAL VALIDATION
# ------------------------------------------------------------

invalid_silver_orders = (
    silver_check
    .join(
        valid_order_ids,
        trim(
            silver_check["order_id"]
            .cast("string")
        )
        == col("_valid_order_id"),
        how="left_anti",
    )
    .count()
)

print(
    "Invalid order IDs in Silver:",
    invalid_silver_orders
)


# ------------------------------------------------------------
# FINAL PAYMENT DATE VALIDATION
# ------------------------------------------------------------

silver_payment_dates = silver_check.select(
    expr(
        "try_to_timestamp(cast(payment_date AS string))"
    ).alias("_payment_date")
)

print(
    "Invalid payment dates in Silver:",
    silver_payment_dates
    .filter(col("_payment_date").isNull())
    .count()
)

print(
    "Future payment dates in Silver:",
    silver_payment_dates
    .filter(
        col("_payment_date")
        > processing_timestamp
    )
    .count()
)


# ------------------------------------------------------------
# FINAL PAYMENT AMOUNT RECONCILIATION
# ------------------------------------------------------------

final_order_reference = (
    orders
    .select(
        trim(
            col("order_id").cast("string")
        ).alias("order_id"),
        expr(
            "try_cast(total_amount AS DOUBLE)"
        ).alias("_order_total"),
    )
)

final_payment_reconciliation = (
    silver_check
    .join(
        final_order_reference,
        on="order_id",
        how="left",
    )
)

invalid_silver_amounts = (
    final_payment_reconciliation
    .filter(
        col("_order_total").isNull()
        |
        (
            expr(
                "round(try_cast(amount AS DOUBLE), 2)"
            )
            != expr(
                "round(_order_total, 2)"
            )
        )
    )
    .count()
)

print(
    "Payment amount mismatches in Silver:",
    invalid_silver_amounts
)


# ============================================================
# FINAL SUCCESS
# ============================================================

print()
print(
    "Final Payment validation completed successfully"
)

spark.stop()