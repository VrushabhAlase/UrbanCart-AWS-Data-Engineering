import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ORDERS_PATH = Path(
    "data/silver/orders/orders.parquet"
)

ORDER_ITEMS_PATH = Path(
    "data/silver/order_items/order_items.parquet"
)

PRODUCTS_PATH = Path(
    "data/silver/products/products.parquet"
)

PAYMENTS_PATH = Path(
    "data/silver/Payments/payments.parquet"
)

GOLD_PATH = Path(
    "data/gold/sales/daily_sales.parquet"
)


# ============================================================
# LOAD SILVER DATA
# ============================================================

orders = pd.read_parquet(
    ORDERS_PATH,
    dtype_backend="numpy_nullable"
)

order_items = pd.read_parquet(
    ORDER_ITEMS_PATH,
    dtype_backend="numpy_nullable"
)

products = pd.read_parquet(
    PRODUCTS_PATH,
    dtype_backend="numpy_nullable"
)

payments = pd.read_parquet(
    PAYMENTS_PATH,
    dtype_backend="numpy_nullable"
)

print("Silver datasets loaded successfully")
print("Orders:", len(orders))
print("Order Items:", len(order_items))
print("Products:", len(products))
print("Payments:", len(payments))


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

orders["order_date"] = pd.to_datetime(
    orders["order_date"],
    errors="coerce"
)

order_items["quantity"] = pd.to_numeric(
    order_items["quantity"],
    errors="coerce"
)

order_items["unit_price"] = pd.to_numeric(
    order_items["unit_price"],
    errors="coerce"
)

order_items["discount_amount"] = pd.to_numeric(
    order_items["discount_amount"],
    errors="coerce"
)

order_items["tax_amount"] = pd.to_numeric(
    order_items["tax_amount"],
    errors="coerce"
)

order_items["item_total"] = pd.to_numeric(
    order_items["item_total"],
    errors="coerce"
)

payments["amount"] = pd.to_numeric(
    payments["amount"],
    errors="coerce"
)

payments["payment_date"] = pd.to_datetime(
    payments["payment_date"],
    errors="coerce"
)


# ============================================================
# SALES ORDER STATUS DEFINITION
# ============================================================

VALID_SALES_STATUSES = {
    "Confirmed",
    "Shipped",
    "Delivered"
}

orders_for_sales = orders[
    orders["order_status"].isin(
        VALID_SALES_STATUSES
    )
].copy()

print()
print("========================================")
print("SALES ORDER FILTER")
print("========================================")

print(
    "Orders included in Sales Gold:",
    len(orders_for_sales)
)

print(
    "Excluded orders:",
    len(orders) - len(orders_for_sales)
)

print()
print("Included statuses:")

print(
    orders_for_sales[
        "order_status"
    ].value_counts()
)


# ============================================================
# ORDER ITEMS + ORDERS
# ============================================================

sales = order_items.merge(
    orders_for_sales[
        [
            "order_id",
            "customer_id",
            "order_date",
            "order_status"
        ]
    ],
    on="order_id",
    how="inner"
)

print()
print(
    "Order Items joined with Orders:",
    len(sales)
)


# ============================================================
# ORDER ITEMS + PRODUCTS
# ============================================================

sales = sales.merge(
    products[
        [
            "product_id",
            "category",
            "subcategory",
            "brand"
        ]
    ],
    on="product_id",
    how="left"
)

print(
    "Sales records after Product join:",
    len(sales)
)


# ============================================================
# SALES DATE
# ============================================================

sales["sales_date"] = (
    sales["order_date"]
    .dt.date
)


# ============================================================
# GROSS ITEM AMOUNT
# ============================================================

sales["gross_item_amount"] = (
    sales["quantity"]
    * sales["unit_price"]
)


# ============================================================
# DAILY SALES AGGREGATION
# ============================================================

daily_sales = (
    sales
    .groupby(
        "sales_date",
        as_index=False
    )
    .agg(
        total_orders=(
            "order_id",
            "nunique"
        ),
        total_order_items=(
            "order_item_id",
            "nunique"
        ),
        total_quantity=(
            "quantity",
            "sum"
        ),
        gross_sales=(
            "gross_item_amount",
            "sum"
        ),
        discount_amount=(
            "discount_amount",
            "sum"
        ),
        tax_amount=(
            "tax_amount",
            "sum"
        ),
        net_sales=(
            "item_total",
            "sum"
        ),
        unique_customers=(
            "customer_id",
            "nunique"
        ),
        unique_products=(
            "product_id",
            "nunique"
        )
    )
)


# ============================================================
# SUCCESSFUL PAYMENT AGGREGATION
# ============================================================

successful_payments = payments[
    payments["payment_status"]
    == "Success"
].copy()

successful_payments[
    "payment_sales_date"
] = (
    successful_payments[
        "payment_date"
    ].dt.date
)

daily_payments = (
    successful_payments
    .groupby(
        "payment_sales_date",
        as_index=False
    )
    .agg(
        successful_payment_amount=(
            "amount",
            "sum"
        )
    )
    .rename(
        columns={
            "payment_sales_date":
                "sales_date"
        }
    )
)


# ============================================================
# ADD PAYMENT METRICS
# ============================================================

daily_sales = daily_sales.merge(
    daily_payments,
    on="sales_date",
    how="left"
)

daily_sales[
    "successful_payment_amount"
] = (
    daily_sales[
        "successful_payment_amount"
    ].fillna(0)
)


# ============================================================
# AVERAGE ORDER VALUE
# ============================================================

daily_sales[
    "average_order_value"
] = (
    daily_sales["net_sales"]
    / daily_sales["total_orders"]
)


# ============================================================
# ROUND FINANCIAL VALUES
# ============================================================

financial_columns = [
    "gross_sales",
    "discount_amount",
    "tax_amount",
    "net_sales",
    "successful_payment_amount",
    "average_order_value"
]

for column in financial_columns:

    daily_sales[column] = (
        daily_sales[column]
        .round(2)
    )


# ============================================================
# SORT
# ============================================================

daily_sales = daily_sales.sort_values(
    "sales_date"
).reset_index(
    drop=True
)


# ============================================================
# CREATE GOLD DIRECTORY
# ============================================================

GOLD_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# WRITE GOLD DATA
# ============================================================

daily_sales.to_parquet(
    GOLD_PATH,
    index=False
)


# ============================================================
# GOLD DATASET SUMMARY
# ============================================================

print()
print("========================================")
print("DAILY SALES GOLD DATASET")
print("========================================")

print(
    "Rows:",
    len(daily_sales)
)

print(
    "Columns:",
    len(daily_sales.columns)
)

print()
print("Columns:")

print(
    daily_sales.columns.tolist()
)

print()
print("Preview:")

print(
    daily_sales.head(10)
)


# ============================================================
# FINAL GOLD VALIDATION
# ============================================================

print()
print("========================================")
print("FINAL DAILY SALES VALIDATION")
print("========================================")

print(
    "Duplicate sales dates:",
    daily_sales[
        "sales_date"
    ].duplicated().sum()
)

print()
print("Null values:")

print(
    daily_sales.isna().sum()
)

print()
print(
    "Negative total orders:",
    (
        daily_sales[
            "total_orders"
        ] < 0
    ).sum()
)

print(
    "Negative total quantity:",
    (
        daily_sales[
            "total_quantity"
        ] < 0
    ).sum()
)

print(
    "Negative net sales:",
    (
        daily_sales[
            "net_sales"
        ] < 0
    ).sum()
)

print()
print(
    "Daily Sales Gold dataset written successfully"
)

print(
    "Output:",
    GOLD_PATH
)