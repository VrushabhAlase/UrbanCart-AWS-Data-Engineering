import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BRONZE_PATH = Path("data/bronze/order_items/order_items.parquet")

SILVER_PATH = Path("data/silver/order_items/order_items.parquet")
DQ_PATH = Path("data/dq/order_items/order_items_dq.parquet")


# ============================================================
# LOAD BRONZE DATA
# ============================================================

df = pd.read_parquet(
    BRONZE_PATH,
    dtype_backend="numpy_nullable"
)

print("Bronze order item data loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# VALID ID RANGES
# ============================================================

# Existing UrbanCart Orders:
# O000001 -> O050000

# Existing UrbanCart Products:
# P000001 -> P010000


# ============================================================
# ORDER ITEM ID VALIDATION
# ============================================================

missing_order_item_id = (
    df["order_item_id"].isna()
    | df["order_item_id"]
    .astype("string")
    .str.strip()
    .eq("")
)

duplicate_order_item_id = df["order_item_id"].duplicated(
    keep="first"
)

print(
    "Missing order item IDs:",
    missing_order_item_id.sum()
)

print(
    "Duplicate order item IDs:",
    duplicate_order_item_id.sum()
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

order_id_pattern = (
    df["order_id"]
    .astype("string")
    .str.fullmatch(r"O\d{6}")
)

order_id_number = pd.to_numeric(
    df["order_id"]
    .astype("string")
    .str[1:],
    errors="coerce"
)

invalid_order_id = (
    ~order_id_pattern.fillna(False)
    | order_id_number.isna()
    | ~order_id_number.between(1, 50000)
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
# PRODUCT ID VALIDATION
# ============================================================

missing_product_id = (
    df["product_id"].isna()
    | df["product_id"]
    .astype("string")
    .str.strip()
    .eq("")
)

product_id_pattern = (
    df["product_id"]
    .astype("string")
    .str.fullmatch(r"P\d{6}")
)

product_id_number = pd.to_numeric(
    df["product_id"]
    .astype("string")
    .str[1:],
    errors="coerce"
)

invalid_product_id = (
    ~product_id_pattern.fillna(False)
    | product_id_number.isna()
    | ~product_id_number.between(1, 10000)
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
# QUANTITY VALIDATION
# ============================================================

quantity = pd.to_numeric(
    df["quantity"],
    errors="coerce"
)

invalid_quantity = (
    quantity.isna()
    | (quantity <= 0)
    | (quantity % 1 != 0)
)

print(
    "Invalid quantities:",
    invalid_quantity.sum()
)


# ============================================================
# UNIT PRICE VALIDATION
# ============================================================

unit_price = pd.to_numeric(
    df["unit_price"],
    errors="coerce"
)

invalid_unit_price = (
    unit_price.isna()
    | (unit_price <= 0)
)

print(
    "Invalid unit prices:",
    invalid_unit_price.sum()
)


# ============================================================
# DISCOUNT VALIDATION
# ============================================================

discount = pd.to_numeric(
    df["discount_amount"],
    errors="coerce"
)

invalid_discount_amount = (
    discount.isna()
    | (discount < 0)
)

print(
    "Invalid discount amounts:",
    invalid_discount_amount.sum()
)


# ============================================================
# DISCOUNT > GROSS AMOUNT
# ============================================================

gross_amount = (
    quantity * unit_price
).round(2)

discount_exceeds_gross = (
    quantity.notna()
    & unit_price.notna()
    & discount.notna()
    & (discount > gross_amount)
)

print(
    "Discount exceeds gross amount:",
    discount_exceeds_gross.sum()
)


# ============================================================
# TAX VALIDATION
# ============================================================

tax = pd.to_numeric(
    df["tax_amount"],
    errors="coerce"
)

invalid_tax_amount = (
    tax.isna()
    | (tax < 0)
)

print(
    "Invalid tax amounts:",
    invalid_tax_amount.sum()
)


# ============================================================
# ITEM TOTAL VALIDATION
# ============================================================

item_total = pd.to_numeric(
    df["item_total"],
    errors="coerce"
)

expected_item_total = (
    quantity
    * unit_price
    - discount
    + tax
).round(2)

invalid_item_total = (
    item_total.isna()
    | (item_total < 0)
    | (
        item_total.notna()
        & expected_item_total.notna()
        & (
            item_total.round(2)
            != expected_item_total
        )
    )
)

print(
    "Invalid item totals:",
    invalid_item_total.sum()
)


# ============================================================
# CREATE DQ ISSUE COLUMN
# ============================================================

df["_dq_issue"] = pd.NA


def add_dq_issue(mask, issue):
    """
    Assign the first detected DQ issue.
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
    duplicate_order_item_id,
    "duplicate_order_item_record"
)

add_dq_issue(
    missing_order_item_id,
    "missing_order_item_id"
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
    missing_product_id,
    "missing_product_id"
)

add_dq_issue(
    invalid_product_id,
    "invalid_product_id"
)

add_dq_issue(
    invalid_quantity,
    "invalid_quantity"
)

add_dq_issue(
    invalid_unit_price,
    "invalid_unit_price"
)

add_dq_issue(
    invalid_discount_amount,
    "invalid_discount_amount"
)

add_dq_issue(
    discount_exceeds_gross,
    "discount_exceeds_gross_amount"
)

add_dq_issue(
    invalid_tax_amount,
    "invalid_tax_amount"
)

add_dq_issue(
    invalid_item_total,
    "invalid_item_total"
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
# CREATE DIRECTORIES
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
# SUMMARY
# ============================================================

print()
print("========================================")
print("ORDER ITEM DATA QUALITY SUMMARY")
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
    "Order Item Silver and DQ data written successfully"
)

# ============================================================
# FINAL SILVER / DQ VALIDATION
# ============================================================

print()
print("========================================")
print("FINAL ORDER ITEM DATA VALIDATION")
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
    "Silver duplicate order item IDs:",
    silver_check["order_item_id"].duplicated().sum()
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
# CUSTOMER ORDER ID RANGE VALIDATION
# ============================================================

silver_order_pattern = (
    silver_check["order_id"]
    .astype("string")
    .str.fullmatch(r"O\d{6}")
)

silver_order_number = pd.to_numeric(
    silver_check["order_id"]
    .astype("string")
    .str[1:],
    errors="coerce"
)

invalid_silver_orders = (
    ~silver_order_pattern.fillna(False)
    | silver_order_number.isna()
    | ~silver_order_number.between(1, 50000)
)

print(
    "Invalid order IDs in Silver:",
    invalid_silver_orders.sum()
)


# ============================================================
# PRODUCT ID RANGE VALIDATION
# ============================================================

silver_product_pattern = (
    silver_check["product_id"]
    .astype("string")
    .str.fullmatch(r"P\d{6}")
)

silver_product_number = pd.to_numeric(
    silver_check["product_id"]
    .astype("string")
    .str[1:],
    errors="coerce"
)

invalid_silver_products = (
    ~silver_product_pattern.fillna(False)
    | silver_product_number.isna()
    | ~silver_product_number.between(1, 10000)
)

print(
    "Invalid product IDs in Silver:",
    invalid_silver_products.sum()
)


# ============================================================
# ITEM TOTAL BUSINESS RULE
# ============================================================

silver_quantity = pd.to_numeric(
    silver_check["quantity"],
    errors="coerce"
)

silver_unit_price = pd.to_numeric(
    silver_check["unit_price"],
    errors="coerce"
)

silver_discount = pd.to_numeric(
    silver_check["discount_amount"],
    errors="coerce"
)

silver_tax = pd.to_numeric(
    silver_check["tax_amount"],
    errors="coerce"
)

silver_item_total = pd.to_numeric(
    silver_check["item_total"],
    errors="coerce"
)

expected_item_total = (
    silver_quantity
    * silver_unit_price
    - silver_discount
    + silver_tax
).round(2)

invalid_silver_item_totals = (
    silver_item_total.round(2)
    != expected_item_total
)

print(
    "Invalid item totals in Silver:",
    invalid_silver_item_totals.sum()
)