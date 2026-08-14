from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


# ============================================================
# 1. PATHS
# ============================================================

RAW_PATH = "data/raw/inventory.parquet"

PRODUCTS_PATH = "data/silver/products/products.parquet"

BRONZE_PATH = "data/bronze/inventory/inventory.parquet"

SILVER_PATH = "data/silver/inventory/inventory.parquet"

DQ_PATH = "data/dq/inventory/inventory_dq.parquet"


# ============================================================
# 2. CONFIGURATION
# ============================================================

PROCESSING_DATE = "2026-08-01"

VALID_WAREHOUSES = [
    "W001",
    "W002",
    "W003",
    "W004",
    "W005",
    "W006",
    "W007",
    "W008",
    "W009",
    "W010",
]

VALID_INVENTORY_STATUSES = [
    "In Stock",
    "Low Stock",
    "Out of Stock",
    "Overstocked",
]

INVENTORY_COLUMNS = [
    "inventory_id",
    "product_id",
    "warehouse_id",
    "stock_quantity",
    "reserved_quantity",
    "available_quantity",
    "reorder_level",
    "inventory_status",
    "last_updated",
]


# ============================================================
# 3. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("UrbanCart-Inventory-Silver")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 4. LOAD RAW INVENTORY
# ============================================================

print("\n========================================")
print("RAW INVENTORY DATA")
print("========================================")

raw_df = spark.read.parquet(RAW_PATH)

print("Rows:", raw_df.count())
print("Columns:", len(raw_df.columns))
print("Column names:", raw_df.columns)

print("\nSchema:")
raw_df.printSchema()

print("\nSample records:")
raw_df.show(10, truncate=False)


# ============================================================
# 5. LOAD PRODUCTS SILVER
# ============================================================

print("\n========================================")
print("PRODUCT REFERENCE DATA")
print("========================================")

products_df = spark.read.parquet(PRODUCTS_PATH)

print(
    "Products Silver rows:",
    products_df.count()
)


# ============================================================
# 6. CREATE VALID PRODUCT ID REFERENCE
# ============================================================

valid_product_ids = (
    products_df
    .select(
        F.trim(
            F.col("product_id").cast("string")
        ).alias("_product_id_ref")
    )
    .filter(
        F.col("_product_id_ref").isNotNull()
        & (
            F.col("_product_id_ref") != ""
        )
    )
    .dropDuplicates(["_product_id_ref"])
)


# ============================================================
# 7. WRITE BRONZE
# ============================================================
#
# IMPORTANT:
# We read RAW and write BRONZE.
#
# We do NOT read BRONZE and overwrite BRONZE.
# This prevents the FAILED_READ_FILE error that occurred
# previously.
# ============================================================

(
    raw_df
    .write
    .mode("overwrite")
    .parquet(BRONZE_PATH)
)

print(
    "\nBronze inventory data written successfully"
)


# ============================================================
# 8. USE RAW DATA FOR TRANSFORMATION
# ============================================================

df = raw_df


# ============================================================
# 9. CLEAN STRING COLUMNS
# ============================================================

df = (
    df
    .withColumn(
        "_inventory_id_clean",
        F.trim(
            F.col("inventory_id").cast("string")
        )
    )
    .withColumn(
        "_product_id_clean",
        F.trim(
            F.col("product_id").cast("string")
        )
    )
    .withColumn(
        "_warehouse_id_clean",
        F.trim(
            F.col("warehouse_id").cast("string")
        )
    )
    .withColumn(
        "_inventory_status_clean",
        F.trim(
            F.col("inventory_status").cast("string")
        )
    )
)


# ============================================================
# 10. DUPLICATE INVENTORY ID
# ============================================================
#
# Original Pandas logic:
#
# duplicated(keep="first")
#
# Therefore only subsequent duplicate rows are invalid.
# ============================================================

duplicate_window = (
    Window
    .partitionBy("_inventory_id_clean")
    .orderBy(
        F.monotonically_increasing_id()
    )
)

df = df.withColumn(
    "_inventory_row_number",
    F.row_number().over(
        duplicate_window
    )
)

duplicate_inventory_id = (
    F.col("_inventory_row_number") > 1
)


# ============================================================
# 11. PRODUCT ID VALIDATION
# ============================================================

missing_product_id = (
    F.col("_product_id_clean").isNull()
    | (
        F.col("_product_id_clean") == ""
    )
)


df = (
    df
    .join(
        F.broadcast(valid_product_ids),
        F.col("_product_id_clean")
        == F.col("_product_id_ref"),
        "left"
    )
    .withColumn(
        "_product_exists",
        F.col("_product_id_ref").isNotNull()
    )
    .drop("_product_id_ref")
)


invalid_product_id = (
    ~F.col("_product_exists")
)


# ============================================================
# 12. INVENTORY ID VALIDATION
# ============================================================

missing_inventory_id = (
    F.col("_inventory_id_clean").isNull()
    | (
        F.col("_inventory_id_clean") == ""
    )
)


# ============================================================
# 13. WAREHOUSE VALIDATION
# ============================================================

invalid_warehouse_id = (
    F.col("_warehouse_id_clean").isNull()
    | ~F.col("_warehouse_id_clean").isin(
        VALID_WAREHOUSES
    )
)


# ============================================================
# 14. NUMERIC CONVERSION
# ============================================================

df = (
    df
    .withColumn(
        "_stock_quantity_num",
        F.col("stock_quantity").cast("double")
    )
    .withColumn(
        "_reserved_quantity_num",
        F.col("reserved_quantity").cast("double")
    )
    .withColumn(
        "_available_quantity_num",
        F.col("available_quantity").cast("double")
    )
    .withColumn(
        "_reorder_level_num",
        F.col("reorder_level").cast("double")
    )
)


# ============================================================
# 15. STOCK QUANTITY VALIDATION
# ============================================================

invalid_stock_quantity = (
    F.col("_stock_quantity_num").isNull()
    | (
        F.col("_stock_quantity_num") < 0
    )
    | (
        F.col("_stock_quantity_num")
        != F.floor(
            F.col("_stock_quantity_num")
        )
    )
)


# ============================================================
# 16. RESERVED QUANTITY VALIDATION
# ============================================================

invalid_reserved_quantity = (
    F.col("_reserved_quantity_num").isNull()
    | (
        F.col("_reserved_quantity_num") < 0
    )
    | (
        F.col("_reserved_quantity_num")
        != F.floor(
            F.col("_reserved_quantity_num")
        )
    )
)


# ============================================================
# 17. RESERVED > STOCK
# ============================================================

reserved_exceeds_stock = (
    F.col("_stock_quantity_num").isNotNull()
    & F.col("_reserved_quantity_num").isNotNull()
    & (
        F.col("_reserved_quantity_num")
        > F.col("_stock_quantity_num")
    )
)


# ============================================================
# 18. AVAILABLE QUANTITY VALIDATION
# ============================================================

invalid_available_quantity = (
    F.col("_available_quantity_num").isNull()
    | (
        F.col("_available_quantity_num") < 0
    )
    | (
        F.col("_available_quantity_num")
        != F.floor(
            F.col("_available_quantity_num")
        )
    )
)


# ============================================================
# 19. AVAILABLE QUANTITY BUSINESS RULE
# ============================================================
#
# available_quantity =
# stock_quantity - reserved_quantity
# ============================================================

available_quantity_mismatch = (
    F.col("_stock_quantity_num").isNotNull()
    & F.col("_reserved_quantity_num").isNotNull()
    & F.col("_available_quantity_num").isNotNull()
    & (
        F.col("_available_quantity_num")
        != (
            F.col("_stock_quantity_num")
            - F.col("_reserved_quantity_num")
        )
    )
)


# ============================================================
# 20. REORDER LEVEL VALIDATION
# ============================================================

invalid_reorder_level = (
    F.col("_reorder_level_num").isNull()
    | (
        F.col("_reorder_level_num") < 0
    )
    | (
        F.col("_reorder_level_num")
        != F.floor(
            F.col("_reorder_level_num")
        )
    )
)


# ============================================================
# 21. INVENTORY STATUS VALIDATION
# ============================================================

invalid_inventory_status = (
    F.col("_inventory_status_clean").isNull()
    | ~F.col(
        "_inventory_status_clean"
    ).isin(
        VALID_INVENTORY_STATUSES
    )
)


# ============================================================
# 22. STATUS BUSINESS RULE
# ============================================================
#
# EXACTLY MATCHES THE ORIGINAL PANDAS LOGIC:
#
# available = 0
#     -> must be Out of Stock
#
# available > 0 AND available <= reorder
#     -> must be Low Stock
#
# available > reorder AND status == Out of Stock
#     -> mismatch
#
# IMPORTANT:
# Every comparison is explicitly parenthesized.
# ============================================================

status_out_of_stock_mismatch = (
    (
        F.col("_available_quantity_num")
        == 0
    )
    & (
        F.col("_inventory_status_clean")
        != "Out of Stock"
    )
)


status_low_stock_mismatch = (
    (
        F.col("_available_quantity_num")
        > 0
    )
    & (
        F.col("_available_quantity_num")
        <= F.col("_reorder_level_num")
    )
    & (
        F.col("_inventory_status_clean")
        != "Low Stock"
    )
)


status_invalid_out_of_stock = (
    (
        F.col("_available_quantity_num")
        > F.col("_reorder_level_num")
    )
    & (
        F.col("_inventory_status_clean")
        == "Out of Stock"
    )
)


status_mismatch = (
    status_out_of_stock_mismatch
    | status_low_stock_mismatch
    | status_invalid_out_of_stock
)


# ============================================================
# 23. LAST UPDATED VALIDATION
# ============================================================
#
# try_to_timestamp is intentionally used.
# It prevents malformed timestamps from crashing the
# entire Spark job.
# ============================================================

df = df.withColumn(
    "_last_updated_ts",
    F.expr(
        "try_to_timestamp(last_updated)"
    )
)


invalid_last_updated = (
    F.col("_last_updated_ts").isNull()
)


future_last_updated = (
    F.col("_last_updated_ts").isNotNull()
    & (
        F.to_date(
            F.col("_last_updated_ts")
        )
        > F.to_date(
            F.lit(PROCESSING_DATE)
        )
    )
)


# ============================================================
# 24. DATA QUALITY SUMMARY COUNTS
# ============================================================

print("\n========================================")
print("INVENTORY DATA QUALITY VALIDATION")
print("========================================")


total_records = df.count()

print(
    "Missing inventory IDs:",
    df.filter(missing_inventory_id).count()
)

print(
    "Duplicate inventory ID rows:",
    df.filter(duplicate_inventory_id).count()
)

print(
    "Missing product IDs:",
    df.filter(missing_product_id).count()
)

print(
    "Invalid product IDs:",
    df.filter(invalid_product_id).count()
)

print(
    "Invalid warehouse IDs:",
    df.filter(invalid_warehouse_id).count()
)

print(
    "Invalid stock quantities:",
    df.filter(invalid_stock_quantity).count()
)

print(
    "Invalid reserved quantities:",
    df.filter(
        invalid_reserved_quantity
    ).count()
)

print(
    "Reserved exceeds stock:",
    df.filter(
        reserved_exceeds_stock
    ).count()
)

print(
    "Invalid available quantities:",
    df.filter(
        invalid_available_quantity
    ).count()
)

print(
    "Available quantity mismatches:",
    df.filter(
        available_quantity_mismatch
    ).count()
)

print(
    "Invalid reorder levels:",
    df.filter(
        invalid_reorder_level
    ).count()
)

print(
    "Invalid inventory statuses:",
    df.filter(
        invalid_inventory_status
    ).count()
)

print(
    "Inventory status mismatches:",
    df.filter(
        status_mismatch
    ).count()
)

print(
    "Invalid last updated dates:",
    df.filter(
        invalid_last_updated
    ).count()
)

print(
    "Future last updated dates:",
    df.filter(
        future_last_updated
    ).count()
)


# ============================================================
# 25. CREATE DQ ISSUE
# ============================================================
#
# Original Pandas code assigns only the FIRST detected
# issue to each record.
#
# Therefore the WHEN order below intentionally matches
# the original add_dq_issue() order.
# ============================================================

df = df.withColumn(
    "_dq_issue",
    F.when(
        duplicate_inventory_id,
        "duplicate_inventory_record"
    )
    .when(
        missing_inventory_id,
        "missing_inventory_id"
    )
    .when(
        missing_product_id,
        "missing_product_id"
    )
    .when(
        invalid_product_id,
        "invalid_product_id"
    )
    .when(
        invalid_warehouse_id,
        "invalid_warehouse_id"
    )
    .when(
        invalid_stock_quantity,
        "invalid_stock_quantity"
    )
    .when(
        invalid_reserved_quantity,
        "invalid_reserved_quantity"
    )
    .when(
        reserved_exceeds_stock,
        "reserved_exceeds_stock"
    )
    .when(
        invalid_available_quantity,
        "invalid_available_quantity"
    )
    .when(
        available_quantity_mismatch,
        "available_quantity_mismatch"
    )
    .when(
        invalid_reorder_level,
        "invalid_reorder_level"
    )
    .when(
        invalid_inventory_status,
        "invalid_inventory_status"
    )
    .when(
        status_mismatch,
        "inventory_status_mismatch"
    )
    .when(
        invalid_last_updated,
        "invalid_last_updated"
    )
    .when(
        future_last_updated,
        "future_last_updated"
    )
)


# ============================================================
# 26. SPLIT SILVER AND DQ
# ============================================================

silver_df = (
    df
    .filter(
        F.col("_dq_issue").isNull()
    )
    .select(
        *INVENTORY_COLUMNS
    )
)


dq_df = (
    df
    .filter(
        F.col("_dq_issue").isNotNull()
    )
    .select(
        *INVENTORY_COLUMNS,
        "_dq_issue"
    )
)


# ============================================================
# 27. DATA QUALITY SUMMARY
# ============================================================

valid_records = silver_df.count()
invalid_records = dq_df.count()

print("\n========================================")
print("INVENTORY DATA QUALITY SUMMARY")
print("========================================")

print(
    "Total records:",
    total_records
)

print(
    "Valid records:",
    valid_records
)

print(
    "Invalid records:",
    invalid_records
)

print("\nDQ issue distribution:")

(
    dq_df
    .groupBy("_dq_issue")
    .count()
    .orderBy(F.desc("count"))
    .show(
        50,
        truncate=False
    )
)


# ============================================================
# 28. SILVER / DQ OUTPUT SUMMARY
# ============================================================

print("\n========================================")
print("INVENTORY SILVER / DQ OUTPUT")
print("========================================")

print(
    "Silver rows:",
    valid_records
)

print(
    "Silver columns:",
    len(silver_df.columns)
)

print(
    "DQ rows:",
    invalid_records
)

print(
    "DQ columns:",
    len(dq_df.columns)
)

print("\nSilver columns:")
print(silver_df.columns)

print("\nDQ columns:")
print(dq_df.columns)


# ============================================================
# 29. WRITE SILVER
# ============================================================

(
    silver_df
    .write
    .mode("overwrite")
    .parquet(SILVER_PATH)
)


# ============================================================
# 30. WRITE DQ
# ============================================================

(
    dq_df
    .write
    .mode("overwrite")
    .parquet(DQ_PATH)
)


print(
    "\nInventory Silver and DQ data written successfully"
)


# ============================================================
# 31. FINAL VALIDATION
# ============================================================

print("\n========================================")
print("FINAL INVENTORY DATA VALIDATION")
print("========================================")


# ------------------------------------------------------------
# Silver duplicate inventory IDs
# ------------------------------------------------------------

silver_duplicate_inventory_ids = (
    silver_df
    .groupBy("inventory_id")
    .count()
    .filter(
        F.col("count") > 1
    )
    .count()
)

print(
    "Silver duplicate inventory IDs:",
    silver_duplicate_inventory_ids
)


# ------------------------------------------------------------
# Silver null values
# ------------------------------------------------------------

print("\nSilver null values:")

silver_null_counts = silver_df.select(
    [
        F.sum(
            F.when(
                F.col(column).isNull(),
                1
            ).otherwise(0)
        ).alias(column)
        for column in INVENTORY_COLUMNS
    ]
)

silver_null_counts.show()


# ------------------------------------------------------------
# DQ issue presence
# ------------------------------------------------------------

print(
    "_dq_issue present in Silver:",
    "_dq_issue" in silver_df.columns
)

print(
    "_dq_issue present in DQ:",
    "_dq_issue" in dq_df.columns
)


# ------------------------------------------------------------
# DQ issue null values
# ------------------------------------------------------------

dq_issue_null_count = (
    dq_df
    .filter(
        F.col("_dq_issue").isNull()
    )
    .count()
)

print(
    "DQ issue null values:",
    dq_issue_null_count
)


# ------------------------------------------------------------
# Invalid product IDs in Silver
# ------------------------------------------------------------

silver_invalid_products = (
    silver_df
    .join(
        F.broadcast(valid_product_ids),
        F.trim(
            F.col("product_id").cast("string")
        )
        == F.col("_product_id_ref"),
        "left_anti"
    )
    .count()
)

print(
    "Invalid product IDs in Silver:",
    silver_invalid_products
)


# ------------------------------------------------------------
# Available quantity validation
# ------------------------------------------------------------

silver_available_mismatch = (
    silver_df
    .filter(
        F.col("available_quantity")
        != (
            F.col("stock_quantity")
            - F.col("reserved_quantity")
        )
    )
    .count()
)

print(
    "Available quantity mismatches in Silver:",
    silver_available_mismatch
)


# ------------------------------------------------------------
# Status validation
# ------------------------------------------------------------

final_status_mismatch = (
    (
        (
            F.col("available_quantity")
            == 0
        )
        & (
            F.col("inventory_status")
            != "Out of Stock"
        )
    )
    |
    (
        (
            F.col("available_quantity")
            > 0
        )
        & (
            F.col("available_quantity")
            <= F.col("reorder_level")
        )
        & (
            F.col("inventory_status")
            != "Low Stock"
        )
    )
    |
    (
        (
            F.col("available_quantity")
            > F.col("reorder_level")
        )
        & (
            F.col("inventory_status")
            == "Out of Stock"
        )
    )
)

silver_status_mismatch = (
    silver_df
    .filter(final_status_mismatch)
    .count()
)

print(
    "Inventory status mismatches in Silver:",
    silver_status_mismatch
)


# ------------------------------------------------------------
# Last updated validation
# ------------------------------------------------------------

silver_future_dates = (
    silver_df
    .filter(
        F.to_date(
            F.expr(
                "try_to_timestamp(last_updated)"
            )
        )
        > F.to_date(
            F.lit(PROCESSING_DATE)
        )
    )
    .count()
)

silver_invalid_dates = (
    silver_df
    .filter(
        F.expr(
            "try_to_timestamp(last_updated)"
        ).isNull()
    )
    .count()
)

print(
    "Invalid last updated dates in Silver:",
    silver_invalid_dates
)

print(
    "Future last updated dates in Silver:",
    silver_future_dates
)


# ============================================================
# 32. FINAL SUCCESS MESSAGE
# ============================================================

print("\n========================================")
print("FINAL INVENTORY VALIDATION COMPLETED")
print("========================================")

print(
    "Inventory Silver and DQ pipeline completed successfully"
)


# ============================================================
# 33. STOP SPARK
# ============================================================

spark.stop()