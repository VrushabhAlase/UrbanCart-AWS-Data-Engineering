from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    when,
    trim,
    row_number,
    try_to_timestamp,
    expr,
    sum as spark_sum,
)
from pyspark.sql.window import Window


# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("UrbanCart-Sellers-PySpark")
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

# Bronze is READ ONLY.
BRONZE_PATH = str(
    PROJECT_ROOT / "data/bronze/sellers/sellers.parquet"
)

SILVER_PATH = str(
    PROJECT_ROOT / "data/silver/pyspark/sellers"
)

DQ_PATH = str(
    PROJECT_ROOT / "data/dq/pyspark/sellers"
)


# ============================================================
# BUSINESS RULES
# ============================================================

VALID_SELLER_STATUSES = [
    "Active",
    "Inactive",
]

EXPECTED_COUNTRY = "India"

PROCESSING_DATE = "2026-08-01"


# ============================================================
# SELLER COLUMNS
# ============================================================

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


# ============================================================
# LOAD BRONZE
# ============================================================

df = spark.read.parquet(BRONZE_PATH)

print()
print("========================================")
print("BRONZE SELLER DATA")
print("========================================")

print("Rows:", df.count())
print("Columns:", len(df.columns))
print("Column names:", df.columns)


# ============================================================
# REMOVE OLD DQ COLUMN
# ============================================================

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
# DATA QUALITY VALIDATION
# ============================================================

print()
print("========================================")
print("SELLER DATA QUALITY VALIDATION")
print("========================================")


# ============================================================
# 1. MISSING SELLER ID
# ============================================================

seller_id_string = trim(
    col("seller_id").cast("string")
)

df = df.withColumn(
    "dq_missing_seller_id",
    col("seller_id").isNull()
    | (
        seller_id_string == ""
    )
)

print(
    "Missing seller IDs:",
    df.filter(
        col("dq_missing_seller_id")
    ).count()
)


# ============================================================
# 2. DUPLICATE SELLER ID
# ============================================================

duplicate_window = (
    Window
    .partitionBy("seller_id")
    .orderBy(lit(1))
)

df = df.withColumn(
    "_seller_id_row_number",
    row_number().over(
        duplicate_window
    )
)

df = df.withColumn(
    "dq_duplicate_seller_id",
    col("_seller_id_row_number") > 1
)

print(
    "Duplicate seller ID rows:",
    df.filter(
        col("dq_duplicate_seller_id")
    ).count()
)


# ============================================================
# 3. MISSING SELLER NAME
# ============================================================

seller_name_string = trim(
    col("seller_name").cast("string")
)

df = df.withColumn(
    "dq_missing_seller_name",
    col("seller_name").isNull()
    | (
        seller_name_string == ""
    )
)

print(
    "Missing seller names:",
    df.filter(
        col("dq_missing_seller_name")
    ).count()
)


# ============================================================
# 4. INVALID EMAIL
# ============================================================

email_string = trim(
    col("email").cast("string")
)

df = df.withColumn(
    "dq_invalid_email",
    col("email").isNull()
    | (
        email_string == ""
    )
    | ~email_string.rlike(
        r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )
)

print(
    "Invalid emails:",
    df.filter(
        col("dq_invalid_email")
    ).count()
)


# ============================================================
# 5. INVALID PHONE
# ============================================================

phone_string = trim(
    col("phone").cast("string")
)

df = df.withColumn(
    "dq_invalid_phone",
    col("phone").isNull()
    | (
        phone_string == ""
    )
    | ~phone_string.rlike(
        r"^[6-9][0-9]{9}$"
    )
)

print(
    "Invalid phones:",
    df.filter(
        col("dq_invalid_phone")
    ).count()
)


# ============================================================
# 6. MISSING CITY
# ============================================================

city_string = trim(
    col("city").cast("string")
)

df = df.withColumn(
    "dq_missing_city",
    col("city").isNull()
    | (
        city_string == ""
    )
)

print(
    "Missing cities:",
    df.filter(
        col("dq_missing_city")
    ).count()
)


# ============================================================
# 7. MISSING STATE
# ============================================================

state_string = trim(
    col("state").cast("string")
)

df = df.withColumn(
    "dq_missing_state",
    col("state").isNull()
    | (
        state_string == ""
    )
)

print(
    "Missing states:",
    df.filter(
        col("dq_missing_state")
    ).count()
)


# ============================================================
# 8. INVALID COUNTRY
# ============================================================

country_string = trim(
    col("country").cast("string")
)

df = df.withColumn(
    "dq_invalid_country",
    col("country").isNull()
    | (
        country_string == ""
    )
    | (
        country_string != EXPECTED_COUNTRY
    )
)

print(
    "Invalid countries:",
    df.filter(
        col("dq_invalid_country")
    ).count()
)


# ============================================================
# 9. INVALID SELLER RATING
# ============================================================

# seller_rating is already numeric in the source.
# try_cast makes this safe if the Bronze schema changes.

seller_rating_number = expr(
    "try_cast(seller_rating AS DOUBLE)"
)

df = df.withColumn(
    "dq_invalid_rating",
    seller_rating_number.isNull()
    | (
        seller_rating_number < 1
    )
    | (
        seller_rating_number > 5
    )
)

print(
    "Invalid seller ratings:",
    df.filter(
        col("dq_invalid_rating")
    ).count()
)


# ============================================================
# 10. INVALID SELLER STATUS
# ============================================================

seller_status_string = trim(
    col("seller_status").cast("string")
)

df = df.withColumn(
    "dq_invalid_status",
    col("seller_status").isNull()
    | (
        seller_status_string == ""
    )
    | ~col("seller_status").isin(
        VALID_SELLER_STATUSES
    )
)

print(
    "Invalid seller statuses:",
    df.filter(
        col("dq_invalid_status")
    ).count()
)


# ============================================================
# 11. INVALID JOINED DATE
# ============================================================

df = df.withColumn(
    "_joined_date_parsed",
    try_to_timestamp(
        trim(
            col("joined_date").cast("string")
        )
    )
)

df = df.withColumn(
    "dq_invalid_joined_date",
    col("_joined_date_parsed").isNull()
)

print(
    "Invalid joined dates:",
    df.filter(
        col("dq_invalid_joined_date")
    ).count()
)


# ============================================================
# 12. FUTURE JOINED DATE
# ============================================================

processing_timestamp = try_to_timestamp(
    lit(PROCESSING_DATE)
)

df = df.withColumn(
    "dq_future_joined_date",
    col("_joined_date_parsed").isNotNull()
    & (
        col("_joined_date_parsed")
        > processing_timestamp
    )
)

print(
    "Future joined dates:",
    df.filter(
        col("dq_future_joined_date")
    ).count()
)


# ============================================================
# CREATE DQ ISSUE
# ============================================================
#
# Preserve the validation rules from transform_sellers.py.
#
# Only the first detected issue is stored in _dq_issue.
#
# ============================================================

df = df.withColumn(
    "_dq_issue",

    when(
        col("dq_missing_seller_id"),
        lit("missing_seller_id")
    )

    .when(
        col("dq_duplicate_seller_id"),
        lit("duplicate_seller_id")
    )

    .when(
        col("dq_missing_seller_name"),
        lit("missing_seller_name")
    )

    .when(
        col("dq_invalid_email"),
        lit("invalid_email")
    )

    .when(
        col("dq_invalid_phone"),
        lit("invalid_phone")
    )

    .when(
        col("dq_missing_city"),
        lit("missing_city")
    )

    .when(
        col("dq_missing_state"),
        lit("missing_state")
    )

    .when(
        col("dq_invalid_country"),
        lit("invalid_country")
    )

    .when(
        col("dq_invalid_rating"),
        lit("invalid_seller_rating")
    )

    .when(
        col("dq_invalid_status"),
        lit("invalid_seller_status")
    )

    .when(
        col("dq_invalid_joined_date"),
        lit("invalid_joined_date")
    )

    .when(
        col("dq_future_joined_date"),
        lit("future_joined_date")
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

    "_seller_id_row_number",

    "_joined_date_parsed",

    "dq_missing_seller_id",
    "dq_duplicate_seller_id",

    "dq_missing_seller_name",

    "dq_invalid_email",
    "dq_invalid_phone",

    "dq_missing_city",
    "dq_missing_state",

    "dq_invalid_country",

    "dq_invalid_rating",

    "dq_invalid_status",

    "dq_invalid_joined_date",
    "dq_future_joined_date",
]


# ============================================================
# SILVER
# ============================================================

silver_df = (
    silver_df
    .select(
        *SELLER_COLUMNS
    )
)


# ============================================================
# DQ
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
# SUMMARY
# ============================================================

print()
print("========================================")
print("SELLER DATA QUALITY SUMMARY")
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
# DQ DISTRIBUTION
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
print("SELLER SILVER / DQ OUTPUT")
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
print("FINAL SELLER VALIDATION")
print("========================================")


# ============================================================
# SILVER DUPLICATE SELLER IDS
# ============================================================

silver_duplicate_count = (
    silver_df
    .groupBy("seller_id")
    .count()
    .filter(
        col("count") > 1
    )
    .count()
)

print(
    "Silver duplicate seller IDs:",
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
# FINAL EMAIL VALIDATION
# ============================================================

invalid_silver_emails = (
    silver_df
    .filter(
        col("email").isNull()
        | (
            trim(
                col("email")
            ) == ""
        )
        | ~trim(
            col("email")
        ).rlike(
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
        )
    )
    .count()
)

print(
    "Invalid emails in Silver:",
    invalid_silver_emails
)


# ============================================================
# FINAL PHONE VALIDATION
# ============================================================

invalid_silver_phones = (
    silver_df
    .filter(
        col("phone").isNull()
        | (
            trim(
                col("phone")
            ) == ""
        )
        | ~trim(
            col("phone")
        ).rlike(
            r"^[6-9][0-9]{9}$"
        )
    )
    .count()
)

print(
    "Invalid phones in Silver:",
    invalid_silver_phones
)


# ============================================================
# FINAL RATING VALIDATION
# ============================================================

invalid_silver_ratings = (
    silver_df
    .filter(
        expr(
            "try_cast(seller_rating AS DOUBLE)"
        ).isNull()
        | (
            expr(
                "try_cast(seller_rating AS DOUBLE)"
            ) < 1
        )
        | (
            expr(
                "try_cast(seller_rating AS DOUBLE)"
            ) > 5
        )
    )
    .count()
)

print(
    "Invalid ratings in Silver:",
    invalid_silver_ratings
)


# ============================================================
# FINAL COUNTRY VALIDATION
# ============================================================

invalid_silver_countries = (
    silver_df
    .filter(
        col("country").isNull()
        | (
            trim(
                col("country")
            ) != EXPECTED_COUNTRY
        )
    )
    .count()
)

print(
    "Invalid countries in Silver:",
    invalid_silver_countries
)


# ============================================================
# FINAL STATUS VALIDATION
# ============================================================

invalid_silver_statuses = (
    silver_df
    .filter(
        col("seller_status").isNull()
        | (
            trim(
                col("seller_status")
            ) == ""
        )
        | ~col("seller_status").isin(
            VALID_SELLER_STATUSES
        )
    )
    .count()
)

print(
    "Invalid statuses in Silver:",
    invalid_silver_statuses
)


# ============================================================
# FINAL JOINED DATE VALIDATION
# ============================================================

silver_joined_date = try_to_timestamp(
    trim(
        col("joined_date").cast("string")
    )
)

invalid_silver_joined_dates = (
    silver_df
    .filter(
        silver_joined_date.isNull()
        | (
            silver_joined_date
            > processing_timestamp
        )
    )
    .count()
)

print(
    "Invalid/future joined dates in Silver:",
    invalid_silver_joined_dates
)


# ============================================================
# DQ COLUMN VALIDATION
# ============================================================

print(
    "_dq_issue present in Silver:",
    "_dq_issue" in silver_df.columns
)

print(
    "_dq_issue present in DQ:",
    "_dq_issue" in dq_df.columns
)


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
# SUCCESS
# ============================================================

print()
print(
    "Seller Silver and DQ data written successfully"
)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()