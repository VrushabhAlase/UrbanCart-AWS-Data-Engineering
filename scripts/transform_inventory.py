import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

RAW_PATH = Path("data/raw/inventory.parquet")

PRODUCTS_PATH = Path(
    "data/silver/products/products.parquet"
)

BRONZE_PATH = Path(
    "data/bronze/inventory/inventory.parquet"
)

SILVER_PATH = Path(
    "data/silver/inventory/inventory.parquet"
)

DQ_PATH = Path(
    "data/dq/inventory/inventory_dq.parquet"
)


# ============================================================
# LOAD RAW INVENTORY
# ============================================================

df = pd.read_parquet(
    RAW_PATH,
    dtype_backend="numpy_nullable"
)

print("Raw inventory data loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Column names:", df.columns.tolist())


# ============================================================
# LOAD PRODUCTS SILVER
# ============================================================

products = pd.read_parquet(
    PRODUCTS_PATH,
    dtype_backend="numpy_nullable"
)

print()
print("Products Silver data loaded successfully")
print("Products rows:", len(products))


# ============================================================
# CREATE BRONZE DIRECTORY
# ============================================================

BRONZE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# WRITE BRONZE
# ============================================================

df.to_parquet(
    BRONZE_PATH,
    index=False
)

print("Bronze inventory data written successfully")


# ============================================================
# VALID VALUES
# ============================================================

VALID_WAREHOUSES = {
    "W001",
    "W002",
    "W003",
    "W004",
    "W005",
    "W006",
    "W007",
    "W008",
    "W009",
    "W010"
}

VALID_INVENTORY_STATUSES = {
    "In Stock",
    "Low Stock",
    "Out of Stock",
    "Overstocked"
}

PROCESSING_DATE = pd.Timestamp(
    "2026-08-01"
)


# ============================================================
# INVENTORY ID VALIDATION
# ============================================================

missing_inventory_id = (
    df["inventory_id"].isna()
    | df["inventory_id"]
    .astype("string")
    .str.strip()
    .eq("")
)

duplicate_inventory_id = (
    df["inventory_id"].duplicated(
        keep="first"
    )
)

print()
print(
    "Missing inventory IDs:",
    missing_inventory_id.sum()
)

print(
    "Duplicate inventory ID rows:",
    duplicate_inventory_id.sum()
)


# ============================================================
# PRODUCT ID VALIDATION
# ============================================================

missing_product_id = (
    df["product_id"].isna()
    | df["product_id"]
    .astype("string")
    .str.strip()
    .eq("")
)

valid_product_ids = set(
    products["product_id"]
    .dropna()
    .astype("string")
)

invalid_product_id = (
    ~df["product_id"]
    .astype("string")
    .isin(valid_product_ids)
)

print(
    "Missing product IDs:",
    missing_product_id.sum()
)

print(
    "Invalid product IDs:",
    invalid_product_id.sum()
)


# ============================================================
# WAREHOUSE VALIDATION
# ============================================================

invalid_warehouse_id = (
    ~df["warehouse_id"].isin(
        VALID_WAREHOUSES
    )
)

print(
    "Invalid warehouse IDs:",
    invalid_warehouse_id.sum()
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

stock_quantity = pd.to_numeric(
    df["stock_quantity"],
    errors="coerce"
)

reserved_quantity = pd.to_numeric(
    df["reserved_quantity"],
    errors="coerce"
)

available_quantity = pd.to_numeric(
    df["available_quantity"],
    errors="coerce"
)

reorder_level = pd.to_numeric(
    df["reorder_level"],
    errors="coerce"
)


# ============================================================
# STOCK QUANTITY VALIDATION
# ============================================================

invalid_stock_quantity = (
    stock_quantity.isna()
    | (stock_quantity < 0)
    | (stock_quantity % 1 != 0)
)

print(
    "Invalid stock quantities:",
    invalid_stock_quantity.sum()
)


# ============================================================
# RESERVED QUANTITY VALIDATION
# ============================================================

invalid_reserved_quantity = (
    reserved_quantity.isna()
    | (reserved_quantity < 0)
    | (reserved_quantity % 1 != 0)
)

print(
    "Invalid reserved quantities:",
    invalid_reserved_quantity.sum()
)


# ============================================================
# RESERVED > STOCK
# ============================================================

reserved_exceeds_stock = (
    stock_quantity.notna()
    & reserved_quantity.notna()
    & (
        reserved_quantity
        > stock_quantity
    )
)

print(
    "Reserved exceeds stock:",
    reserved_exceeds_stock.sum()
)


# ============================================================
# AVAILABLE QUANTITY VALIDATION
# ============================================================

invalid_available_quantity = (
    available_quantity.isna()
    | (available_quantity < 0)
    | (available_quantity % 1 != 0)
)

print(
    "Invalid available quantities:",
    invalid_available_quantity.sum()
)


# ============================================================
# AVAILABLE QUANTITY BUSINESS RULE
# ============================================================

expected_available_quantity = (
    stock_quantity
    - reserved_quantity
)

available_quantity_mismatch = (
    stock_quantity.notna()
    & reserved_quantity.notna()
    & available_quantity.notna()
    & (
        available_quantity
        != expected_available_quantity
    )
)

print(
    "Available quantity mismatches:",
    available_quantity_mismatch.sum()
)


# ============================================================
# REORDER LEVEL
# ============================================================

invalid_reorder_level = (
    reorder_level.isna()
    | (reorder_level < 0)
    | (reorder_level % 1 != 0)
)

print(
    "Invalid reorder levels:",
    invalid_reorder_level.sum()
)


# ============================================================
# INVENTORY STATUS
# ============================================================

invalid_inventory_status = (
    ~df["inventory_status"].isin(
        VALID_INVENTORY_STATUSES
    )
)

print(
    "Invalid inventory statuses:",
    invalid_inventory_status.sum()
)


# ============================================================
# STATUS BUSINESS RULE
# ============================================================

status_mismatch = (
    (
        (available_quantity == 0)
        & (
            df["inventory_status"]
            != "Out of Stock"
        )
    )
    |
    (
        (available_quantity > 0)
        & (
            available_quantity
            <= reorder_level
        )
        & (
            df["inventory_status"]
            != "Low Stock"
        )
    )
    |
    (
        (available_quantity > reorder_level)
        & (
            df["inventory_status"]
            == "Out of Stock"
        )
    )
)

print(
    "Inventory status mismatches:",
    status_mismatch.sum()
)


# ============================================================
# LAST UPDATED VALIDATION
# ============================================================

last_updated = pd.to_datetime(
    df["last_updated"],
    errors="coerce"
)

invalid_last_updated = (
    last_updated.isna()
)

future_last_updated = (
    last_updated.notna()
    & (
        last_updated
        > PROCESSING_DATE
    )
)

print(
    "Invalid last updated dates:",
    invalid_last_updated.sum()
)

print(
    "Future last updated dates:",
    future_last_updated.sum()
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
    duplicate_inventory_id,
    "duplicate_inventory_record"
)

add_dq_issue(
    missing_inventory_id,
    "missing_inventory_id"
)

add_dq_issue(
    missing_product_id,
    "missing_product_id"
)

add_dq_issue(
    invalid_product_id,
    "invalid_product_id"
)

add_dq_issue(
    invalid_warehouse_id,
    "invalid_warehouse_id"
)

add_dq_issue(
    invalid_stock_quantity,
    "invalid_stock_quantity"
)

add_dq_issue(
    invalid_reserved_quantity,
    "invalid_reserved_quantity"
)

add_dq_issue(
    reserved_exceeds_stock,
    "reserved_exceeds_stock"
)

add_dq_issue(
    invalid_available_quantity,
    "invalid_available_quantity"
)

add_dq_issue(
    available_quantity_mismatch,
    "available_quantity_mismatch"
)

add_dq_issue(
    invalid_reorder_level,
    "invalid_reorder_level"
)

add_dq_issue(
    invalid_inventory_status,
    "invalid_inventory_status"
)

add_dq_issue(
    status_mismatch,
    "inventory_status_mismatch"
)

add_dq_issue(
    invalid_last_updated,
    "invalid_last_updated"
)

add_dq_issue(
    future_last_updated,
    "future_last_updated"
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
print("INVENTORY DATA QUALITY SUMMARY")
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
    "Inventory Silver and DQ data written successfully"
)


# ============================================================
# FINAL SILVER / DQ VALIDATION
# ============================================================

print()
print("========================================")
print("FINAL INVENTORY DATA VALIDATION")
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
    "Silver duplicate inventory IDs:",
    silver_check[
        "inventory_id"
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
# FINAL PRODUCT REFERENTIAL VALIDATION
# ============================================================

silver_invalid_products = (
    ~silver_check[
        "product_id"
    ]
    .astype("string")
    .isin(valid_product_ids)
)

print(
    "Invalid product IDs in Silver:",
    silver_invalid_products.sum()
)


# ============================================================
# FINAL AVAILABLE QUANTITY VALIDATION
# ============================================================

final_stock = pd.to_numeric(
    silver_check[
        "stock_quantity"
    ],
    errors="coerce"
)

final_reserved = pd.to_numeric(
    silver_check[
        "reserved_quantity"
    ],
    errors="coerce"
)

final_available = pd.to_numeric(
    silver_check[
        "available_quantity"
    ],
    errors="coerce"
)

final_expected_available = (
    final_stock
    - final_reserved
)

invalid_silver_available = (
    final_available
    != final_expected_available
)

print(
    "Available quantity mismatches in Silver:",
    invalid_silver_available.sum()
)


# ============================================================
# FINAL STATUS VALIDATION
# ============================================================

final_reorder = pd.to_numeric(
    silver_check[
        "reorder_level"
    ],
    errors="coerce"
)

final_status_mismatch = (
    (
        (final_available == 0)
        & (
            silver_check[
                "inventory_status"
            ]
            != "Out of Stock"
        )
    )
    |
    (
        (final_available > 0)
        & (
            final_available
            <= final_reorder
        )
        & (
            silver_check[
                "inventory_status"
            ]
            != "Low Stock"
        )
    )
    |
    (
        (final_available > final_reorder)
        & (
            silver_check[
                "inventory_status"
            ]
            == "Out of Stock"
        )
    )
)

print(
    "Inventory status mismatches in Silver:",
    final_status_mismatch.sum()
)


# ============================================================
# FINAL DATE VALIDATION
# ============================================================

final_last_updated = pd.to_datetime(
    silver_check[
        "last_updated"
    ],
    errors="coerce"
)

print(
    "Invalid last updated dates in Silver:",
    final_last_updated.isna().sum()
)

print(
    "Future last updated dates in Silver:",
    (
        final_last_updated
        > PROCESSING_DATE
    ).sum()
)


# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print()
print(
    "Final Inventory validation completed successfully"
)