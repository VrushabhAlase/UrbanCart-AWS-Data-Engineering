import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BRONZE_PATH = Path("data/bronze/products/products.parquet")

SILVER_PATH = Path("data/silver/products/products.parquet")
DQ_PATH = Path("data/dq/products/products_dq.parquet")


# ============================================================
# LOAD BRONZE DATA
# ============================================================

df = pd.read_parquet(
    BRONZE_PATH,
    dtype_backend="numpy_nullable"
)

print("Bronze product data loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# VALID VALUES / BUSINESS RULES
# ============================================================

VALID_CATEGORIES = {
    "Electronics": [
        "Mobiles",
        "Laptops",
        "Headphones",
        "Cameras",
        "Smart Watches",
        "Accessories"
    ],

    "Fashion": [
        "Men's Clothing",
        "Women's Clothing",
        "Footwear",
        "Watches",
        "Accessories"
    ],

    "Home & Kitchen": [
        "Cookware",
        "Furniture",
        "Home Decor",
        "Storage",
        "Kitchen Appliances"
    ],

    "Beauty & Personal Care": [
        "Skincare",
        "Haircare",
        "Makeup",
        "Fragrances",
        "Personal Hygiene"
    ],

    "Grocery": [
        "Staples",
        "Snacks",
        "Beverages",
        "Dairy",
        "Personal Care"
    ],

    "Sports & Fitness": [
        "Fitness Equipment",
        "Sportswear",
        "Outdoor Gear",
        "Yoga"
    ],

    "Books": [
        "Fiction",
        "Non-Fiction",
        "Academic",
        "Children's Books",
        "Comics"
    ],

    "Toys": [
        "Action Figures",
        "Educational Toys",
        "Board Games",
        "Outdoor Toys",
        "Puzzles"
    ],

    "Automotive": [
        "Car Accessories",
        "Bike Accessories",
        "Car Care",
        "Tools"
    ],

    "Health & Wellness": [
        "Supplements",
        "Medical Devices",
        "Ayurveda",
        "Personal Health"
    ]
}

VALID_STATUSES = [
    "Active",
    "Inactive",
    "Discontinued"
]


# ============================================================
# BASIC VALIDATIONS
# ============================================================

# 1. Missing Product ID
missing_product_id = (
    df["product_id"].isna()
    | df["product_id"].astype("string").str.strip().eq("")
)

print(
    "Missing product IDs:",
    missing_product_id.sum()
)


# 2. Duplicate Product ID
duplicate_product_id = df["product_id"].duplicated(
    keep="first"
)

print(
    "Duplicate product ID rows:",
    duplicate_product_id.sum()
)


# 3. Invalid Seller ID
seller_id_pattern = (
    df["seller_id"]
    .astype("string")
    .str.fullmatch(r"S\d{5}")
)

seller_id_number = pd.to_numeric(
    df["seller_id"].astype("string").str[1:],
    errors="coerce"
)

invalid_seller_id = (
    ~seller_id_pattern.fillna(False)
    | seller_id_number.isna()
    | ~seller_id_number.between(1, 1000)
)

print(
    "Invalid seller IDs:",
    invalid_seller_id.sum()
)


# 4. Missing Product Name
missing_product_name = (
    df["product_name"].isna()
    | df["product_name"].astype("string").str.strip().eq("")
)

print(
    "Missing product names:",
    missing_product_name.sum()
)


# 5. Invalid Category
invalid_category = ~df["category"].isin(
    VALID_CATEGORIES.keys()
)

print(
    "Invalid categories:",
    invalid_category.sum()
)


# 6. Invalid Subcategory
invalid_subcategory = pd.Series(
    False,
    index=df.index
)

for category, subcategories in VALID_CATEGORIES.items():

    category_mask = df["category"].eq(category)

    invalid_subcategory |= (
        category_mask
        & ~df["subcategory"].isin(subcategories)
    )

# If category itself is invalid, treat the row as invalid
invalid_subcategory |= invalid_category

print(
    "Invalid subcategories:",
    invalid_subcategory.sum()
)


# 7. Invalid Unit Price
unit_price_numeric = pd.to_numeric(
    df["unit_price"],
    errors="coerce"
)

invalid_unit_price = (
    unit_price_numeric.isna()
    | (unit_price_numeric <= 0)
)

print(
    "Invalid unit prices:",
    invalid_unit_price.sum()
)


# 8. Invalid Cost Price
cost_price_numeric = pd.to_numeric(
    df["cost_price"],
    errors="coerce"
)

invalid_cost_price = (
    cost_price_numeric.isna()
    | (cost_price_numeric <= 0)
    | (
        cost_price_numeric
        >= unit_price_numeric
    )
)

print(
    "Invalid cost prices:",
    invalid_cost_price.sum()
)


# 9. Invalid Stock Quantity
stock_quantity_numeric = pd.to_numeric(
    df["stock_quantity"],
    errors="coerce"
)

invalid_stock_quantity = (
    stock_quantity_numeric.isna()
    | (stock_quantity_numeric < 0)
)

print(
    "Invalid stock quantities:",
    invalid_stock_quantity.sum()
)


# 10. Invalid Product Status
invalid_product_status = ~df["product_status"].isin(
    VALID_STATUSES
)

print(
    "Invalid product statuses:",
    invalid_product_status.sum()
)


# ============================================================
# DATE VALIDATION
# ============================================================

created_date = pd.to_datetime(
    df["created_date"],
    errors="coerce"
)

# 11. Invalid Created Date
invalid_created_date = created_date.isna()

print(
    "Invalid created dates:",
    invalid_created_date.sum()
)


# 12. Future Created Date

PROCESSING_DATE = pd.Timestamp("2026-08-01")

future_created_date = (
    created_date.notna()
    & (created_date > PROCESSING_DATE)
)

print(
    "Future created dates:",
    future_created_date.sum()
)


# ============================================================
# CREATE DQ ISSUE
# ============================================================

df["_dq_issue"] = pd.NA


def add_dq_issue(mask, issue):
    """
    Add a DQ issue only if the row does not already
    have an issue.
    """
    available = mask & df["_dq_issue"].isna()
    df.loc[available, "_dq_issue"] = issue


# Order matters.
# The first detected issue becomes the primary DQ reason.

add_dq_issue(
    duplicate_product_id,
    "duplicate_product_record"
)

add_dq_issue(
    missing_product_id,
    "missing_required_field_product_id"
)

add_dq_issue(
    invalid_seller_id,
    "invalid_seller_id"
)

add_dq_issue(
    missing_product_name,
    "missing_product_name"
)

add_dq_issue(
    invalid_category,
    "invalid_category"
)

add_dq_issue(
    invalid_subcategory,
    "invalid_subcategory"
)

add_dq_issue(
    invalid_unit_price,
    "invalid_unit_price"
)

add_dq_issue(
    invalid_cost_price,
    "invalid_cost_price"
)

add_dq_issue(
    invalid_stock_quantity,
    "invalid_stock_quantity"
)

add_dq_issue(
    invalid_product_status,
    "invalid_product_status"
)

add_dq_issue(
    invalid_created_date,
    "invalid_created_date"
)

add_dq_issue(
    future_created_date,
    "future_created_date"
)


# ============================================================
# SPLIT VALID / INVALID RECORDS
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
print("PRODUCT DATA QUALITY SUMMARY")
print("========================================")

print("Total records:", len(df))
print("Valid records:", len(silver_df))
print("Invalid records:", len(dq_df))

print()
print("Silver rows:", len(silver_df))
print("Silver columns:", len(silver_df.columns))

print()
print("DQ rows:", len(dq_df))
print("DQ columns:", len(dq_df.columns))

print()
print("Silver columns:")
print(silver_df.columns.tolist())

print()
print("DQ columns:")
print(dq_df.columns.tolist())

print()
print("DQ issue distribution:")
print(
    dq_df["_dq_issue"]
    .value_counts(dropna=False)
)

print()
print("Silver customer DQ column present:",
      "_dq_issue" in silver_df.columns)

print()
print("Product Silver and DQ data written successfully")

# ============================================================
# FINAL SILVER / DQ VALIDATION
# ============================================================

print()
print("========================================")
print("FINAL PRODUCT DATA VALIDATION")
print("========================================")

# Reload the files we just created
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
    "Silver duplicate product IDs:",
    silver_check["product_id"].duplicated().sum()
)

print(
    "Silver null values:"
)

print(
    silver_check.isna().sum()
)

print(
    "_dq_issue present in Silver:",
    "_dq_issue" in silver_check.columns
)

print(
    "DQ rows:", len(dq_check)
)

print(
    "_dq_issue present in DQ:",
    "_dq_issue" in dq_check.columns
)

print(
    "DQ issue null values:",
    dq_check["_dq_issue"].isna().sum()
)

# Validate seller IDs in Silver
silver_seller_numbers = pd.to_numeric(
    silver_check["seller_id"].astype("string").str[1:],
    errors="coerce"
)

invalid_silver_sellers = (
    ~silver_check["seller_id"]
    .astype("string")
    .str.fullmatch(r"S\d{5}")
    | ~silver_seller_numbers.between(1, 1000)
)

print(
    "Invalid seller IDs in Silver:",
    invalid_silver_sellers.sum()
)