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
    .appName("UrbanCart-Customers-PySpark")
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
    PROJECT_ROOT / "data/bronze/customers/customers.parquet"
)

SILVER_PATH = str(
    PROJECT_ROOT / "data/silver/pyspark/customers"
)

DQ_PATH = str(
    PROJECT_ROOT / "data/dq/pyspark/customers"
)


# ============================================================
# BUSINESS RULES
# ============================================================

VALID_GENDERS = [
    "Male",
    "Female",
    "Other",
]

VALID_LOYALTY_TIERS = [
    "Bronze",
    "Silver",
    "Gold",
]

VALID_CUSTOMER_STATUSES = [
    "Active",
    "Inactive",
]

EXPECTED_COUNTRY = "India"


# ============================================================
# PROCESSING DATE
# ============================================================

PROCESSING_DATE = "2026-08-01"


# ============================================================
# CUSTOMER COLUMNS
# ============================================================

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


# ============================================================
# LOAD BRONZE
# ============================================================

df = spark.read.parquet(BRONZE_PATH)

print()
print("========================================")
print("BRONZE CUSTOMER DATA")
print("========================================")

print("Rows:", df.count())
print("Columns:", len(df.columns))
print("Column names:", df.columns)


# ============================================================
# REMOVE EXISTING DQ COLUMN
# ============================================================

# The raw/bronze dataset already contains _dq_issue.
# We calculate DQ again in this Spark pipeline.

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
print("CUSTOMER DATA QUALITY VALIDATION")
print("========================================")


# ============================================================
# 1. MISSING CUSTOMER ID
# ============================================================

customer_id_string = trim(
    col("customer_id").cast("string")
)

df = df.withColumn(
    "dq_missing_customer_id",
    col("customer_id").isNull()
    | (
        customer_id_string == ""
    )
)

print(
    "Missing customer IDs:",
    df.filter(
        col("dq_missing_customer_id")
    ).count()
)


# ============================================================
# 2. DUPLICATE CUSTOMER ID
# ============================================================

duplicate_window = (
    Window
    .partitionBy("customer_id")
    .orderBy(lit(1))
)

df = df.withColumn(
    "_customer_id_row_number",
    row_number().over(
        duplicate_window
    )
)

df = df.withColumn(
    "dq_duplicate_customer_id",
    col("_customer_id_row_number") > 1
)

print(
    "Duplicate customer ID rows:",
    df.filter(
        col("dq_duplicate_customer_id")
    ).count()
)


# ============================================================
# 3. MISSING FIRST NAME
# ============================================================

first_name_string = trim(
    col("first_name").cast("string")
)

df = df.withColumn(
    "dq_missing_first_name",
    col("first_name").isNull()
    | (
        first_name_string == ""
    )
)

print(
    "Missing first names:",
    df.filter(
        col("dq_missing_first_name")
    ).count()
)


# ============================================================
# 4. MISSING LAST NAME
# ============================================================

last_name_string = trim(
    col("last_name").cast("string")
)

df = df.withColumn(
    "dq_missing_last_name",
    col("last_name").isNull()
    | (
        last_name_string == ""
    )
)

print(
    "Missing last names:",
    df.filter(
        col("dq_missing_last_name")
    ).count()
)


# ============================================================
# 5. INVALID GENDER
# ============================================================

gender_string = trim(
    col("gender").cast("string")
)

df = df.withColumn(
    "dq_invalid_gender",
    col("gender").isNull()
    | (
        gender_string == ""
    )
    | ~col("gender").isin(
        VALID_GENDERS
    )
)

print(
    "Invalid gender:",
    df.filter(
        col("dq_invalid_gender")
    ).count()
)


# ============================================================
# 6. INVALID DATE OF BIRTH
# ============================================================

df = df.withColumn(
    "_date_of_birth_parsed",
    try_to_timestamp(
        trim(
            col("date_of_birth").cast("string")
        )
    )
)

df = df.withColumn(
    "dq_invalid_date_of_birth",
    col("_date_of_birth_parsed").isNull()
)

print(
    "Invalid date of birth:",
    df.filter(
        col("dq_invalid_date_of_birth")
    ).count()
)


# ============================================================
# 7. INVALID EMAIL
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
    "Invalid email:",
    df.filter(
        col("dq_invalid_email")
    ).count()
)


# ============================================================
# 8. INVALID PHONE
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
    "Invalid phone:",
    df.filter(
        col("dq_invalid_phone")
    ).count()
)


# ============================================================
# 9. MISSING CITY
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
    "Missing city:",
    df.filter(
        col("dq_missing_city")
    ).count()
)


# ============================================================
# 10. MISSING STATE
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
    "Missing state:",
    df.filter(
        col("dq_missing_state")
    ).count()
)


# ============================================================
# 11. INVALID COUNTRY
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
    "Invalid country:",
    df.filter(
        col("dq_invalid_country")
    ).count()
)


# ============================================================
# 12. INVALID POSTAL CODE
# ============================================================

postal_code_string = trim(
    col("postal_code").cast("string")
)

df = df.withColumn(
    "dq_invalid_postal_code",
    col("postal_code").isNull()
    | (
        postal_code_string == ""
    )
    | ~postal_code_string.rlike(
        r"^[0-9]{6}$"
    )
)

print(
    "Invalid postal codes:",
    df.filter(
        col("dq_invalid_postal_code")
    ).count()
)


# ============================================================
# 13. INVALID REGISTRATION DATE
# ============================================================

df = df.withColumn(
    "_registration_date_parsed",
    try_to_timestamp(
        trim(
            col("registration_date").cast("string")
        )
    )
)

df = df.withColumn(
    "dq_invalid_registration_date",
    col("_registration_date_parsed").isNull()
)

print(
    "Invalid registration dates:",
    df.filter(
        col("dq_invalid_registration_date")
    ).count()
)


# ============================================================
# 14. INVALID LOYALTY TIER
# ============================================================

loyalty_tier_string = trim(
    col("loyalty_tier").cast("string")
)

df = df.withColumn(
    "dq_invalid_loyalty_tier",
    col("loyalty_tier").isNull()
    | (
        loyalty_tier_string == ""
    )
    | ~col("loyalty_tier").isin(
        VALID_LOYALTY_TIERS
    )
)

print(
    "Invalid loyalty tiers:",
    df.filter(
        col("dq_invalid_loyalty_tier")
    ).count()
)


# ============================================================
# 15. INVALID CUSTOMER STATUS
# ============================================================

customer_status_string = trim(
    col("customer_status").cast("string")
)

df = df.withColumn(
    "dq_invalid_customer_status",
    col("customer_status").isNull()
    | (
        customer_status_string == ""
    )
    | ~col("customer_status").isin(
        VALID_CUSTOMER_STATUSES
    )
)

print(
    "Invalid customer status:",
    df.filter(
        col("dq_invalid_customer_status")
    ).count()
)


# ============================================================
# FUTURE DATE CHECKS
# ============================================================

processing_timestamp = try_to_timestamp(
    lit(PROCESSING_DATE)
)


# ============================================================
# 16. FUTURE DATE OF BIRTH
# ============================================================

df = df.withColumn(
    "dq_future_date_of_birth",
    col("_date_of_birth_parsed").isNotNull()
    & (
        col("_date_of_birth_parsed")
        > processing_timestamp
    )
)


# ============================================================
# 17. FUTURE REGISTRATION DATE
# ============================================================

df = df.withColumn(
    "dq_future_registration_date",
    col("_registration_date_parsed").isNotNull()
    & (
        col("_registration_date_parsed")
        > processing_timestamp
    )
)


# ============================================================
# CREATE DQ ISSUE
# ============================================================
#
# We preserve the existing Python validation rules.
#
# Only the FIRST detected issue is stored in _dq_issue.
#
# ============================================================

df = df.withColumn(
    "_dq_issue",

    when(
        col("dq_missing_customer_id"),
        lit("missing_customer_id")
    )

    .when(
        col("dq_duplicate_customer_id"),
        lit("duplicate_customer_id")
    )

    .when(
        col("dq_missing_first_name"),
        lit("missing_first_name")
    )

    .when(
        col("dq_missing_last_name"),
        lit("missing_last_name")
    )

    .when(
        col("dq_invalid_gender"),
        lit("invalid_gender")
    )

    .when(
        col("dq_invalid_date_of_birth"),
        lit("invalid_date_of_birth")
    )

    .when(
        col("dq_future_date_of_birth"),
        lit("future_date_of_birth")
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
        col("dq_invalid_postal_code"),
        lit("invalid_postal_code")
    )

    .when(
        col("dq_invalid_registration_date"),
        lit("invalid_registration_date")
    )

    .when(
        col("dq_future_registration_date"),
        lit("future_registration_date")
    )

    .when(
        col("dq_invalid_loyalty_tier"),
        lit("invalid_loyalty_tier")
    )

    .when(
        col("dq_invalid_customer_status"),
        lit("invalid_customer_status")
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

    "_customer_id_row_number",

    "_date_of_birth_parsed",

    "_registration_date_parsed",

    "dq_missing_customer_id",
    "dq_duplicate_customer_id",

    "dq_missing_first_name",
    "dq_missing_last_name",

    "dq_invalid_gender",

    "dq_invalid_date_of_birth",
    "dq_future_date_of_birth",

    "dq_invalid_email",
    "dq_invalid_phone",

    "dq_missing_city",
    "dq_missing_state",

    "dq_invalid_country",
    "dq_invalid_postal_code",

    "dq_invalid_registration_date",
    "dq_future_registration_date",

    "dq_invalid_loyalty_tier",
    "dq_invalid_customer_status",
]


# ============================================================
# SILVER DATA
# ============================================================

silver_df = (
    silver_df
    .select(
        *CUSTOMER_COLUMNS
    )
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
# SUMMARY
# ============================================================

print()
print("========================================")
print("CUSTOMER DATA QUALITY SUMMARY")
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
print("CUSTOMER SILVER / DQ OUTPUT")
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
print("FINAL CUSTOMER VALIDATION")
print("========================================")


# ============================================================
# SILVER DUPLICATE CUSTOMER IDs
# ============================================================

silver_duplicate_count = (
    silver_df
    .groupBy("customer_id")
    .count()
    .filter(
        col("count") > 1
    )
    .count()
)

print(
    "Silver duplicate customer IDs:",
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
# SILVER EMAIL VALIDATION
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
# SILVER PHONE VALIDATION
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
# SILVER POSTAL CODE VALIDATION
# ============================================================

invalid_silver_postal_codes = (
    silver_df
    .filter(
        col("postal_code").isNull()
        | (
            trim(
                col("postal_code")
            ) == ""
        )
        | ~trim(
            col("postal_code")
        ).rlike(
            r"^[0-9]{6}$"
        )
    )
    .count()
)

print(
    "Invalid postal codes in Silver:",
    invalid_silver_postal_codes
)


# ============================================================
# SILVER COUNTRY VALIDATION
# ============================================================

invalid_silver_country = (
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
    invalid_silver_country
)


# ============================================================
# DQ ISSUE VALIDATION
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
# FINAL SUCCESS
# ============================================================

print()
print(
    "Customer Silver and DQ data written successfully"
)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()