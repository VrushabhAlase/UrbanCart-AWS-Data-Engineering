from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    trim,
    when,
    row_number,
    expr,
    round as spark_round,
    sum as spark_sum,
)
from pyspark.sql.window import Window


# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("UrbanCart-OrderItems-PySpark")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/home/vrual/UrbanCart-AWS-Data-Engineering"
)

BRONZE_PATH = str(
    PROJECT_ROOT / "data/bronze/order_items/order_items.parquet"
)

SILVER_PATH = str(
    PROJECT_ROOT / "data/silver/pyspark/order_items"
)

DQ_PATH = str(
    PROJECT_ROOT / "data/dq/pyspark/order_items"
)


# ============================================================
# VALID ID RANGES
# ============================================================

MIN_ORDER_ID = 1
MAX_ORDER_ID = 50000

MIN_PRODUCT_ID = 1
MAX_PRODUCT_ID = 10000


# ============================================================
# LOAD BRONZE
# ============================================================

df = spark.read.parquet(BRONZE_PATH)

print()
print("========================================")
print("BRONZE ORDER ITEM DATA")
print("========================================")

print("Rows:", df.count())
print("Columns:", len(df.columns))
print("Column names:", df.columns)


# ============================================================
# REMOVE EXISTING DQ COLUMN
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
print("ORDER ITEM DATA QUALITY VALIDATION")
print("========================================")


# ============================================================
# 1. MISSING ORDER ITEM ID
# ============================================================

order_item_id_string = trim(
    col("order_item_id").cast("string")
)

df = df.withColumn(
    "dq_missing_order_item_id",
    col("order_item_id").isNull()
    | (
        order_item_id_string == ""
    )
)

print(
    "Missing order item IDs:",
    df.filter(
        col("dq_missing_order_item_id")
    ).count()
)


# ============================================================
# 2. DUPLICATE ORDER ITEM ID
# ============================================================
#
# IMPORTANT:
# Existing Pandas logic uses:
#
# duplicated(keep="first")
#
# Therefore only the SECOND occurrence onward is marked
# as duplicate.
# ============================================================

duplicate_window = (
    Window
    .partitionBy("order_item_id")
    .orderBy(lit(1))
)

df = df.withColumn(
    "_order_item_row_number",
    row_number().over(duplicate_window)
)

df = df.withColumn(
    "dq_duplicate_order_item_id",
    col("_order_item_row_number") > 1
)

print(
    "Duplicate order item IDs:",
    df.filter(
        col("dq_duplicate_order_item_id")
    ).count()
)


# ============================================================
# 3. MISSING ORDER ID
# ============================================================

order_id_string = trim(
    col("order_id").cast("string")
)

df = df.withColumn(
    "dq_missing_order_id",
    col("order_id").isNull()
    | (
        order_id_string == ""
    )
)

print(
    "Missing order IDs:",
    df.filter(
        col("dq_missing_order_id")
    ).count()
)


# ============================================================
# 4. INVALID ORDER ID
# ============================================================

order_id_number = expr(
    """
    try_cast(
        substring(trim(order_id), 2, 20)
        AS BIGINT
    )
    """
)

valid_order_id_pattern = (
    order_id_string.rlike(
        r"^O[0-9]{6}$"
    )
)

df = df.withColumn(
    "dq_invalid_order_id",
    ~valid_order_id_pattern
    | order_id_number.isNull()
    | (
        order_id_number < MIN_ORDER_ID
    )
    | (
        order_id_number > MAX_ORDER_ID
    )
)

print(
    "Invalid order IDs:",
    df.filter(
        col("dq_invalid_order_id")
    ).count()
)


# ============================================================
# 5. MISSING PRODUCT ID
# ============================================================

product_id_string = trim(
    col("product_id").cast("string")
)

df = df.withColumn(
    "dq_missing_product_id",
    col("product_id").isNull()
    | (
        product_id_string == ""
    )
)

print(
    "Missing product IDs:",
    df.filter(
        col("dq_missing_product_id")
    ).count()
)


# ============================================================
# 6. INVALID PRODUCT ID
# ============================================================

product_id_number = expr(
    """
    try_cast(
        substring(trim(product_id), 2, 20)
        AS BIGINT
    )
    """
)

valid_product_id_pattern = (
    product_id_string.rlike(
        r"^P[0-9]{6}$"
    )
)

df = df.withColumn(
    "dq_invalid_product_id",
    ~valid_product_id_pattern
    | product_id_number.isNull()
    | (
        product_id_number < MIN_PRODUCT_ID
    )
    | (
        product_id_number > MAX_PRODUCT_ID
    )
)

print(
    "Invalid product IDs:",
    df.filter(
        col("dq_invalid_product_id")
    ).count()
)


# ============================================================
# 7. QUANTITY
# ============================================================

quantity_number = expr(
    "try_cast(quantity AS DOUBLE)"
)

df = df.withColumn(
    "_quantity_number",
    quantity_number
)

df = df.withColumn(
    "dq_invalid_quantity",
    col("_quantity_number").isNull()
    | (
        col("_quantity_number") <= 0
    )
    | (
        col("_quantity_number")
        != expr("floor(_quantity_number)")
    )
)

print(
    "Invalid quantities:",
    df.filter(
        col("dq_invalid_quantity")
    ).count()
)


# ============================================================
# 8. UNIT PRICE
# ============================================================

unit_price_number = expr(
    "try_cast(unit_price AS DOUBLE)"
)

df = df.withColumn(
    "_unit_price_number",
    unit_price_number
)

df = df.withColumn(
    "dq_invalid_unit_price",
    col("_unit_price_number").isNull()
    | (
        col("_unit_price_number") <= 0
    )
)

print(
    "Invalid unit prices:",
    df.filter(
        col("dq_invalid_unit_price")
    ).count()
)


# ============================================================
# 9. DISCOUNT
# ============================================================

discount_number = expr(
    "try_cast(discount_amount AS DOUBLE)"
)

df = df.withColumn(
    "_discount_number",
    discount_number
)

df = df.withColumn(
    "dq_invalid_discount_amount",
    col("_discount_number").isNull()
    | (
        col("_discount_number") < 0
    )
)

print(
    "Invalid discount amounts:",
    df.filter(
        col("dq_invalid_discount_amount")
    ).count()
)


# ============================================================
# 10. DISCOUNT > GROSS AMOUNT
# ============================================================

df = df.withColumn(
    "_gross_amount",
    spark_round(
        col("_quantity_number")
        * col("_unit_price_number"),
        2
    )
)

df = df.withColumn(
    "dq_discount_exceeds_gross",
    col("_quantity_number").isNotNull()
    & col("_unit_price_number").isNotNull()
    & col("_discount_number").isNotNull()
    & (
        col("_discount_number")
        > col("_gross_amount")
    )
)

print(
    "Discount exceeds gross amount:",
    df.filter(
        col("dq_discount_exceeds_gross")
    ).count()
)


# ============================================================
# 11. TAX
# ============================================================

tax_number = expr(
    "try_cast(tax_amount AS DOUBLE)"
)

df = df.withColumn(
    "_tax_number",
    tax_number
)

df = df.withColumn(
    "dq_invalid_tax_amount",
    col("_tax_number").isNull()
    | (
        col("_tax_number") < 0
    )
)

print(
    "Invalid tax amounts:",
    df.filter(
        col("dq_invalid_tax_amount")
    ).count()
)


# ============================================================
# 12. ITEM TOTAL
# ============================================================

item_total_number = expr(
    "try_cast(item_total AS DOUBLE)"
)

df = df.withColumn(
    "_item_total_number",
    item_total_number
)

df = df.withColumn(
    "_expected_item_total",
    spark_round(
        col("_quantity_number")
        * col("_unit_price_number")
        - col("_discount_number")
        + col("_tax_number"),
        2
    )
)

df = df.withColumn(
    "dq_invalid_item_total",
    col("_item_total_number").isNull()
    | (
        col("_item_total_number") < 0
    )
    | (
        col("_item_total_number").isNotNull()
        & col("_expected_item_total").isNotNull()
        & (
            spark_round(
                col("_item_total_number"),
                2
            )
            != col("_expected_item_total")
        )
    )
)

print(
    "Invalid item totals:",
    df.filter(
        col("dq_invalid_item_total")
    ).count()
)


# ============================================================
# CREATE DQ ISSUE
# ============================================================
#
# Same order as existing Pandas implementation.
# First detected issue wins.
# ============================================================

df = df.withColumn(
    "_dq_issue",

    when(
        col("dq_duplicate_order_item_id"),
        lit("duplicate_order_item_record")
    )

    .when(
        col("dq_missing_order_item_id"),
        lit("missing_order_item_id")
    )

    .when(
        col("dq_missing_order_id"),
        lit("missing_order_id")
    )

    .when(
        col("dq_invalid_order_id"),
        lit("invalid_order_id")
    )

    .when(
        col("dq_missing_product_id"),
        lit("missing_product_id")
    )

    .when(
        col("dq_invalid_product_id"),
        lit("invalid_product_id")
    )

    .when(
        col("dq_invalid_quantity"),
        lit("invalid_quantity")
    )

    .when(
        col("dq_invalid_unit_price"),
        lit("invalid_unit_price")
    )

    .when(
        col("dq_invalid_discount_amount"),
        lit("invalid_discount_amount")
    )

    .when(
        col("dq_discount_exceeds_gross"),
        lit("discount_exceeds_gross_amount")
    )

    .when(
        col("dq_invalid_tax_amount"),
        lit("invalid_tax_amount")
    )

    .when(
        col("dq_invalid_item_total"),
        lit("invalid_item_total")
    )
)


# ============================================================
# SPLIT SILVER / DQ
# ============================================================

silver_df = df.filter(
    col("_dq_issue").isNull()
)

dq_df = df.filter(
    col("_dq_issue").isNotNull()
)


# ============================================================
# REMOVE HELPER COLUMNS FROM SILVER
# ============================================================

silver_columns = [
    "order_item_id",
    "order_id",
    "product_id",
    "quantity",
    "unit_price",
    "discount_amount",
    "tax_amount",
    "item_total",
]

silver_df = silver_df.select(
    *silver_columns
)


# ============================================================
# CLEAN DQ DATA
# ============================================================

helper_columns = [
    "_order_item_row_number",
    "_quantity_number",
    "_unit_price_number",
    "_discount_number",
    "_gross_amount",
    "_tax_number",
    "_item_total_number",
    "_expected_item_total",

    "dq_missing_order_item_id",
    "dq_duplicate_order_item_id",
    "dq_missing_order_id",
    "dq_invalid_order_id",
    "dq_missing_product_id",
    "dq_invalid_product_id",
    "dq_invalid_quantity",
    "dq_invalid_unit_price",
    "dq_invalid_discount_amount",
    "dq_discount_exceeds_gross",
    "dq_invalid_tax_amount",
    "dq_invalid_item_total",
]

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
print("ORDER ITEM DATA QUALITY SUMMARY")
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
print("ORDER ITEM SILVER / DQ OUTPUT")
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

print()
print("Silver columns:")
print(
    silver_df.columns
)

print()
print("DQ columns:")
print(
    dq_df.columns
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("========================================")
print("FINAL ORDER ITEM DATA VALIDATION")
print("========================================")


# ============================================================
# DUPLICATE ORDER ITEM IDs
# ============================================================

silver_duplicate_count = (
    silver_df
    .groupBy("order_item_id")
    .count()
    .filter(
        col("count") > 1
    )
    .count()
)

print(
    "Silver duplicate order item IDs:",
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
# DQ COLUMN CHECK
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
# ORDER ID RANGE VALIDATION
# ============================================================

silver_order_string = trim(
    col("order_id").cast("string")
)

silver_order_number = expr(
    """
    try_cast(
        substring(trim(order_id), 2, 20)
        AS BIGINT
    )
    """
)

invalid_silver_orders = (
    ~silver_order_string.rlike(
        r"^O[0-9]{6}$"
    )
    | silver_order_number.isNull()
    | (
        silver_order_number < MIN_ORDER_ID
    )
    | (
        silver_order_number > MAX_ORDER_ID
    )
)

print(
    "Invalid order IDs in Silver:",
    silver_df.filter(
        invalid_silver_orders
    ).count()
)


# ============================================================
# PRODUCT ID RANGE VALIDATION
# ============================================================

silver_product_string = trim(
    col("product_id").cast("string")
)

silver_product_number = expr(
    """
    try_cast(
        substring(trim(product_id), 2, 20)
        AS BIGINT
    )
    """
)

invalid_silver_products = (
    ~silver_product_string.rlike(
        r"^P[0-9]{6}$"
    )
    | silver_product_number.isNull()
    | (
        silver_product_number < MIN_PRODUCT_ID
    )
    | (
        silver_product_number > MAX_PRODUCT_ID
    )
)

print(
    "Invalid product IDs in Silver:",
    silver_df.filter(
        invalid_silver_products
    ).count()
)


# ============================================================
# QUANTITY VALIDATION
# ============================================================

invalid_silver_quantity = (
    expr(
        "try_cast(quantity AS DOUBLE)"
    ).isNull()
    | (
        expr(
            "try_cast(quantity AS DOUBLE)"
        ) <= 0
    )
    | (
        expr(
            "try_cast(quantity AS DOUBLE)"
        )
        !=
        expr(
            "floor(try_cast(quantity AS DOUBLE))"
        )
    )
)

print(
    "Invalid quantities in Silver:",
    silver_df.filter(
        invalid_silver_quantity
    ).count()
)


# ============================================================
# UNIT PRICE VALIDATION
# ============================================================

invalid_silver_unit_price = (
    expr(
        "try_cast(unit_price AS DOUBLE)"
    ).isNull()
    | (
        expr(
            "try_cast(unit_price AS DOUBLE)"
        ) <= 0
    )
)

print(
    "Invalid unit prices in Silver:",
    silver_df.filter(
        invalid_silver_unit_price
    ).count()
)


# ============================================================
# DISCOUNT VALIDATION
# ============================================================

invalid_silver_discount = (
    expr(
        "try_cast(discount_amount AS DOUBLE)"
    ).isNull()
    | (
        expr(
            "try_cast(discount_amount AS DOUBLE)"
        ) < 0
    )
)

print(
    "Invalid discount amounts in Silver:",
    silver_df.filter(
        invalid_silver_discount
    ).count()
)


# ============================================================
# TAX VALIDATION
# ============================================================

invalid_silver_tax = (
    expr(
        "try_cast(tax_amount AS DOUBLE)"
    ).isNull()
    | (
        expr(
            "try_cast(tax_amount AS DOUBLE)"
        ) < 0
    )
)

print(
    "Invalid tax amounts in Silver:",
    silver_df.filter(
        invalid_silver_tax
    ).count()
)


# ============================================================
# ITEM TOTAL BUSINESS RULE
# ============================================================

silver_expected_item_total = spark_round(
    expr(
        """
        try_cast(quantity AS DOUBLE)
        *
        try_cast(unit_price AS DOUBLE)
        -
        try_cast(discount_amount AS DOUBLE)
        +
        try_cast(tax_amount AS DOUBLE)
        """
    ),
    2
)

invalid_silver_item_total = (
    expr(
        "try_cast(item_total AS DOUBLE)"
    ).isNull()
    | (
        expr(
            "try_cast(item_total AS DOUBLE)"
        ) < 0
    )
    | (
        expr(
            "try_cast(item_total AS DOUBLE)"
        ).isNotNull()
        & silver_expected_item_total.isNotNull()
        & (
            spark_round(
                expr(
                    "try_cast(item_total AS DOUBLE)"
                ),
                2
            )
            != silver_expected_item_total
        )
    )
)

print(
    "Invalid item totals in Silver:",
    silver_df.filter(
        invalid_silver_item_total
    ).count()
)


# ============================================================
# SUCCESS
# ============================================================

print()
print(
    "Order Item Silver and DQ data written successfully"
)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()