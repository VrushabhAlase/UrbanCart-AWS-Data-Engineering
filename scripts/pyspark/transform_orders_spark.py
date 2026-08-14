from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    when,
    trim,
    row_number,
    try_to_timestamp,
    round as spark_round,
    sum as spark_sum,
    expr,
)
from pyspark.sql.window import Window


# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("UrbanCart-Orders-PySpark")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/home/vrual/UrbanCart-AWS-Data-Engineering"
)

# IMPORTANT:
# Bronze is READ ONLY in this Silver pipeline.
# We do NOT overwrite Bronze here.

BRONZE_PATH = str(
    PROJECT_ROOT / "data/bronze/orders/orders.parquet"
)

SILVER_PATH = str(
    PROJECT_ROOT / "data/silver/pyspark/orders"
)

DQ_PATH = str(
    PROJECT_ROOT / "data/dq/pyspark/orders"
)


# ============================================================
# VALID VALUES
# ============================================================

VALID_ORDER_STATUSES = [
    "Pending",
    "Confirmed",
    "Shipped",
    "Delivered",
    "Cancelled",
    "Returned",
]

VALID_PAYMENT_METHODS = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Net Banking",
    "Cash on Delivery",
    "Wallet",
]


# ============================================================
# PROCESSING DATE
# ============================================================

PROCESSING_DATE = "2026-08-01"


# ============================================================
# LOAD BRONZE
# ============================================================

df = spark.read.parquet(BRONZE_PATH)

print()
print("========================================")
print("BRONZE ORDER DATA")
print("========================================")

print("Rows:", df.count())
print("Columns:", len(df.columns))
print("Column names:", df.columns)


# ============================================================
# REMOVE OLD DQ COLUMN
# ============================================================

# Raw/Bronze data already contains _dq_issue.
# Silver processing must calculate its own DQ result.

if "_dq_issue" in df.columns:
    df = df.drop("_dq_issue")


# ============================================================
# SCHEMA
# ============================================================

print()
print("Schema:")

df.printSchema()


# ============================================================
# SAMPLE
# ============================================================

print()
print("Sample records:")

df.show(
    10,
    truncate=False
)


# ============================================================
# IMPORTANT
# ============================================================
#
# DO NOT WRITE BACK TO BRONZE.
#
# Bronze is the input/source for this Silver pipeline.
#
# ============================================================


# ============================================================
# DATA QUALITY VALIDATION
# ============================================================

print()
print("========================================")
print("ORDER DATA QUALITY VALIDATION")
print("========================================")


# ============================================================
# 1. MISSING ORDER ID
# ============================================================

df = df.withColumn(
    "dq_missing_order_id",
    col("order_id").isNull()
    | (
        trim(
            col("order_id").cast("string")
        ) == ""
    )
)


# ============================================================
# 2. DUPLICATE ORDER ID
# ============================================================

duplicate_window = (
    Window
    .partitionBy("order_id")
    .orderBy(lit(1))
)

df = df.withColumn(
    "_order_id_row_number",
    row_number().over(
        duplicate_window
    )
)

df = df.withColumn(
    "dq_duplicate_order_id",
    col("_order_id_row_number") > 1
)


# ============================================================
# CUSTOMER ID VALIDATION
# ============================================================

customer_id_string = trim(
    col("customer_id").cast("string")
)

# SAFE CAST
#
# Invalid values such as:
# ""
# ABC123
# CXXXXXX
#
# become NULL instead of crashing Spark.

customer_id_number = expr(
    "try_cast(substring(customer_id, 2, 20) AS BIGINT)"
)


# ============================================================
# 3. MISSING CUSTOMER ID
# ============================================================

df = df.withColumn(
    "dq_missing_customer_id",
    col("customer_id").isNull()
    | (
        customer_id_string == ""
    )
)


# ============================================================
# 4. INVALID CUSTOMER ID
# ============================================================

# Expected:
# C000001 through C050000

df = df.withColumn(
    "dq_invalid_customer_id",
    (
        ~customer_id_string.rlike(
            r"^C[0-9]{6}$"
        )
    )
    | customer_id_number.isNull()
    | ~customer_id_number.between(
        1,
        50000
    )
)


# ============================================================
# ORDER DATE VALIDATION
# ============================================================

df = df.withColumn(
    "_order_date_parsed",
    try_to_timestamp(
        trim(
            col("order_date").cast("string")
        )
    )
)


# ============================================================
# 5. INVALID ORDER DATE
# ============================================================

df = df.withColumn(
    "dq_invalid_order_date",
    col("_order_date_parsed").isNull()
)


# ============================================================
# 6. FUTURE ORDER DATE
# ============================================================

processing_timestamp = (
    try_to_timestamp(
        lit(PROCESSING_DATE)
    )
)

df = df.withColumn(
    "dq_future_order_date",
    col("_order_date_parsed").isNotNull()
    & (
        col("_order_date_parsed")
        > processing_timestamp
    )
)


# ============================================================
# 7. INVALID ORDER STATUS
# ============================================================

df = df.withColumn(
    "dq_invalid_order_status",
    col("order_status").isNull()
    | (
        trim(
            col("order_status").cast("string")
        ) == ""
    )
    | ~col("order_status").isin(
        VALID_ORDER_STATUSES
    )
)


# ============================================================
# 8. INVALID PAYMENT METHOD
# ============================================================

df = df.withColumn(
    "dq_invalid_payment_method",
    col("payment_method").isNull()
    | (
        trim(
            col("payment_method").cast("string")
        ) == ""
    )
    | ~col("payment_method").isin(
        VALID_PAYMENT_METHODS
    )
)


# ============================================================
# 9. MISSING SHIPPING CITY
# ============================================================

df = df.withColumn(
    "dq_missing_shipping_city",
    col("shipping_city").isNull()
    | (
        trim(
            col("shipping_city").cast("string")
        ) == ""
    )
)


# ============================================================
# 10. MISSING SHIPPING STATE
# ============================================================

df = df.withColumn(
    "dq_missing_shipping_state",
    col("shipping_state").isNull()
    | (
        trim(
            col("shipping_state").cast("string")
        ) == ""
    )
)


# ============================================================
# 11. INVALID SHIPPING POSTAL CODE
# ============================================================

shipping_postal_string = trim(
    col("shipping_postal_code").cast("string")
)

df = df.withColumn(
    "dq_invalid_shipping_postal_code",
    shipping_postal_string.isNull()
    | (
        shipping_postal_string == ""
    )
    | ~shipping_postal_string.rlike(
        r"^[0-9]{6}$"
    )
)


# ============================================================
# AMOUNT VALIDATION
# ============================================================
#
# IMPORTANT:
# Use try_cast instead of normal cast.
#
# Spark 4.2 ANSI mode throws an exception for values such
# as empty strings. try_cast converts them to NULL so that
# the DQ rules can handle them correctly.
#
# ============================================================

df = (
    df
    .withColumn(
        "_subtotal",
        expr(
            "try_cast(subtotal_amount AS DOUBLE)"
        )
    )
    .withColumn(
        "_discount",
        expr(
            "try_cast(discount_amount AS DOUBLE)"
        )
    )
    .withColumn(
        "_shipping_fee",
        expr(
            "try_cast(shipping_fee AS DOUBLE)"
        )
    )
    .withColumn(
        "_tax",
        expr(
            "try_cast(tax_amount AS DOUBLE)"
        )
    )
    .withColumn(
        "_total",
        expr(
            "try_cast(total_amount AS DOUBLE)"
        )
    )
)


# ============================================================
# 12. INVALID SUBTOTAL
# ============================================================

df = df.withColumn(
    "dq_invalid_subtotal_amount",
    col("_subtotal").isNull()
    | (
        col("_subtotal") <= 0
    )
)


# ============================================================
# 13. INVALID DISCOUNT
# ============================================================

df = df.withColumn(
    "dq_invalid_discount_amount",
    col("_discount").isNull()
    | (
        col("_discount") < 0
    )
)


# ============================================================
# 14. DISCOUNT GREATER THAN SUBTOTAL
# ============================================================

df = df.withColumn(
    "dq_discount_exceeds_subtotal",
    col("_subtotal").isNotNull()
    & col("_discount").isNotNull()
    & (
        col("_discount")
        > col("_subtotal")
    )
)


# ============================================================
# 15. INVALID SHIPPING FEE
# ============================================================

df = df.withColumn(
    "dq_invalid_shipping_fee",
    col("_shipping_fee").isNull()
    | (
        col("_shipping_fee") < 0
    )
)


# ============================================================
# 16. INVALID TAX
# ============================================================

df = df.withColumn(
    "dq_invalid_tax_amount",
    col("_tax").isNull()
    | (
        col("_tax") < 0
    )
)


# ============================================================
# TOTAL AMOUNT BUSINESS RULE
# ============================================================
#
# Expected:
#
# subtotal
# - discount
# + shipping fee
# + tax
# = total
#
# ============================================================

df = df.withColumn(
    "_expected_total",
    spark_round(
        col("_subtotal")
        - col("_discount")
        + col("_shipping_fee")
        + col("_tax"),
        2
    )
)

df = df.withColumn(
    "_actual_total",
    spark_round(
        col("_total"),
        2
    )
)


# ============================================================
# 17. INVALID TOTAL AMOUNT
# ============================================================

df = df.withColumn(
    "dq_invalid_total_amount",
    col("_total").isNull()
    | (
        col("_total") < 0
    )
    | (
        col("_total").isNotNull()
        & col("_expected_total").isNotNull()
        & (
            col("_actual_total")
            != col("_expected_total")
        )
    )
)


# ============================================================
# CREATE DQ ISSUE
# ============================================================
#
# IMPORTANT:
# Existing business logic assigns ONLY THE FIRST
# detected DQ issue.
#
# Priority is preserved from the existing pipeline.
#
# ============================================================

df = df.withColumn(
    "_dq_issue",

    when(
        col("dq_duplicate_order_id"),
        lit("duplicate_order_record")
    )

    .when(
        col("dq_missing_order_id"),
        lit("missing_required_field_order_id")
    )

    .when(
        col("dq_missing_customer_id"),
        lit("missing_customer_id")
    )

    .when(
        col("dq_invalid_customer_id"),
        lit("invalid_customer_id")
    )

    .when(
        col("dq_invalid_order_date"),
        lit("invalid_order_date")
    )

    .when(
        col("dq_future_order_date"),
        lit("future_order_date")
    )

    .when(
        col("dq_invalid_order_status"),
        lit("invalid_order_status")
    )

    .when(
        col("dq_invalid_payment_method"),
        lit("invalid_payment_method")
    )

    .when(
        col("dq_missing_shipping_city"),
        lit("missing_shipping_city")
    )

    .when(
        col("dq_missing_shipping_state"),
        lit("missing_shipping_state")
    )

    .when(
        col("dq_invalid_shipping_postal_code"),
        lit("invalid_shipping_postal_code")
    )

    .when(
        col("dq_invalid_subtotal_amount"),
        lit("invalid_subtotal_amount")
    )

    .when(
        col("dq_invalid_discount_amount"),
        lit("invalid_discount_amount")
    )

    .when(
        col("dq_discount_exceeds_subtotal"),
        lit("discount_exceeds_subtotal")
    )

    .when(
        col("dq_invalid_shipping_fee"),
        lit("invalid_shipping_fee")
    )

    .when(
        col("dq_invalid_tax_amount"),
        lit("invalid_tax_amount")
    )

    .when(
        col("dq_invalid_total_amount"),
        lit("invalid_total_amount")
    )
)


# ============================================================
# SPLIT SILVER AND DQ
# ============================================================

silver_df = df.filter(
    col("_dq_issue").isNull()
)

dq_df = df.filter(
    col("_dq_issue").isNotNull()
)


# ============================================================
# HELPER COLUMNS
# ============================================================

helper_columns = [

    "_order_id_row_number",

    "_order_date_parsed",

    "_subtotal",
    "_discount",
    "_shipping_fee",
    "_tax",
    "_total",

    "_expected_total",
    "_actual_total",

    "dq_missing_order_id",
    "dq_duplicate_order_id",

    "dq_missing_customer_id",
    "dq_invalid_customer_id",

    "dq_invalid_order_date",
    "dq_future_order_date",

    "dq_invalid_order_status",
    "dq_invalid_payment_method",

    "dq_missing_shipping_city",
    "dq_missing_shipping_state",
    "dq_invalid_shipping_postal_code",

    "dq_invalid_subtotal_amount",
    "dq_invalid_discount_amount",
    "dq_discount_exceeds_subtotal",

    "dq_invalid_shipping_fee",
    "dq_invalid_tax_amount",
    "dq_invalid_total_amount",
]


# ============================================================
# CLEAN SILVER DATA
# ============================================================

silver_df = silver_df.drop(
    *helper_columns,
    "_dq_issue"
)


# ============================================================
# DQ DATA
# ============================================================

dq_df = dq_df.drop(
    *helper_columns
)


# ============================================================
# COUNTS
# ============================================================

total_records = df.count()

silver_records = silver_df.count()

dq_records = dq_df.count()


# ============================================================
# DQ SUMMARY
# ============================================================

print()
print("========================================")
print("ORDER DATA QUALITY SUMMARY")
print("========================================")

print(
    "Total records:",
    total_records
)

print(
    "Valid records:",
    silver_records
)

print(
    "Invalid records:",
    dq_records
)


# ============================================================
# DQ ISSUE DISTRIBUTION
# ============================================================

print()
print("DQ issue distribution:")

(
    dq_df
    .groupBy("_dq_issue")
    .count()
    .orderBy(
        col("count").desc()
    )
    .show(
        truncate=False
    )
)


# ============================================================
# WRITE SILVER
# ============================================================

(
    silver_df.write
    .mode("overwrite")
    .parquet(SILVER_PATH)
)


# ============================================================
# WRITE DQ
# ============================================================

(
    dq_df.write
    .mode("overwrite")
    .parquet(DQ_PATH)
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

print()
print("========================================")
print("ORDER SILVER / DQ OUTPUT")
print("========================================")

print(
    "Silver rows:",
    silver_records
)

print(
    "Silver columns:",
    len(silver_df.columns)
)

print(
    "DQ rows:",
    dq_records
)

print(
    "DQ columns:",
    len(dq_df.columns)
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("========================================")
print("FINAL ORDER DATA VALIDATION")
print("========================================")


# ============================================================
# SILVER DUPLICATE ORDER IDs
# ============================================================

silver_duplicate_count = (
    silver_df
    .groupBy("order_id")
    .count()
    .filter(
        col("count") > 1
    )
    .count()
)

print(
    "Silver duplicate order IDs:",
    silver_duplicate_count
)


# ============================================================
# SILVER NULL VALUES
# ============================================================

print()
print("Silver null values:")

silver_df.select(
    [
        spark_sum(
            col(c)
            .isNull()
            .cast("int")
        ).alias(c)
        for c in silver_df.columns
    ]
).show()


# ============================================================
# DQ COLUMN CHECKS
# ============================================================

print(
    "_dq_issue present in Silver:",
    "_dq_issue" in silver_df.columns
)

print(
    "_dq_issue present in DQ:",
    "_dq_issue" in dq_df.columns
)


# ============================================================
# DQ ISSUE NULL CHECK
# ============================================================

dq_issue_null_count = (
    dq_df
    .filter(
        col("_dq_issue").isNull()
    )
    .count()
)

print(
    "DQ issue null values:",
    dq_issue_null_count
)


# ============================================================
# FINAL CUSTOMER ID VALIDATION
# ============================================================

silver_customer_string = trim(
    col("customer_id").cast("string")
)

silver_customer_number = expr(
    "try_cast(substring(customer_id, 2, 20) AS BIGINT)"
)

invalid_silver_customers = (
    silver_customer_string.isNull()
    | (
        silver_customer_string == ""
    )
    | ~silver_customer_string.rlike(
        r"^C[0-9]{6}$"
    )
    | silver_customer_number.isNull()
    | ~silver_customer_number.between(
        1,
        50000
    )
)

invalid_silver_customer_count = (
    silver_df
    .filter(
        invalid_silver_customers
    )
    .count()
)

print(
    "Invalid customer IDs in Silver:",
    invalid_silver_customer_count
)


# ============================================================
# FINAL TOTAL AMOUNT VALIDATION
# ============================================================

silver_validation_df = (
    silver_df

    .withColumn(
        "_subtotal_check",
        expr(
            "try_cast(subtotal_amount AS DOUBLE)"
        )
    )

    .withColumn(
        "_discount_check",
        expr(
            "try_cast(discount_amount AS DOUBLE)"
        )
    )

    .withColumn(
        "_shipping_check",
        expr(
            "try_cast(shipping_fee AS DOUBLE)"
        )
    )

    .withColumn(
        "_tax_check",
        expr(
            "try_cast(tax_amount AS DOUBLE)"
        )
    )

    .withColumn(
        "_total_check",
        expr(
            "try_cast(total_amount AS DOUBLE)"
        )
    )

    .withColumn(
        "_expected_total_check",
        spark_round(
            col("_subtotal_check")
            - col("_discount_check")
            + col("_shipping_check")
            + col("_tax_check"),
            2
        )
    )

    .withColumn(
        "_actual_total_check",
        spark_round(
            col("_total_check"),
            2
        )
    )
)


invalid_silver_totals = (
    silver_validation_df
    .filter(
        col("_total_check").isNull()
        | (
            col("_actual_total_check")
            != col("_expected_total_check")
        )
    )
    .count()
)

print(
    "Invalid total amounts in Silver:",
    invalid_silver_totals
)


# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print()
print(
    "Order Silver and DQ data written successfully"
)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()