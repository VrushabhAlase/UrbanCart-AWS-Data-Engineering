import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CUSTOMERS_PATH = Path(
    "data/silver/customers/customers.parquet"
)

ORDERS_PATH = Path(
    "data/silver/orders/orders.parquet"
)

ORDER_ITEMS_PATH = Path(
    "data/silver/order_items/order_items.parquet"
)

PAYMENTS_PATH = Path(
    "data/silver/Payments/payments.parquet"
)

GOLD_PATH = Path(
    "data/gold/customer/customer_360.parquet"
)


# ============================================================
# LOAD SILVER DATA
# ============================================================

customers = pd.read_parquet(
    CUSTOMERS_PATH,
    dtype_backend="numpy_nullable"
)

orders = pd.read_parquet(
    ORDERS_PATH,
    dtype_backend="numpy_nullable"
)

order_items = pd.read_parquet(
    ORDER_ITEMS_PATH,
    dtype_backend="numpy_nullable"
)

payments = pd.read_parquet(
    PAYMENTS_PATH,
    dtype_backend="numpy_nullable"
)


print("Silver datasets loaded successfully")

print("Customers:", len(customers))
print("Orders:", len(orders))
print("Order Items:", len(order_items))
print("Payments:", len(payments))


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

customers["registration_date"] = pd.to_datetime(
    customers["registration_date"],
    errors="coerce"
)

orders["order_date"] = pd.to_datetime(
    orders["order_date"],
    errors="coerce"
)

order_items["quantity"] = pd.to_numeric(
    order_items["quantity"],
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
# ORDER STATUS DEFINITIONS
# ============================================================

COMPLETED_STATUSES = {
    "Confirmed",
    "Shipped",
    "Delivered"
}


# ============================================================
# CUSTOMER BASE
# ============================================================

customer_360 = customers[
    [
        "customer_id",
        "first_name",
        "last_name",
        "city",
        "state",
        "country",
        "loyalty_tier",
        "customer_status",
        "registration_date"
    ]
].copy()


# ============================================================
# ORDER COUNTS BY CUSTOMER
# ============================================================

order_counts = (
    orders
    .groupby(
        "customer_id",
        as_index=False
    )
    .agg(
        total_orders=(
            "order_id",
            "nunique"
        )
    )
)


customer_360 = customer_360.merge(
    order_counts,
    on="customer_id",
    how="left"
)


# ============================================================
# COMPLETED ORDERS
# ============================================================

completed_orders = orders[
    orders["order_status"].isin(
        COMPLETED_STATUSES
    )
].copy()


completed_order_counts = (
    completed_orders
    .groupby(
        "customer_id",
        as_index=False
    )
    .agg(
        completed_orders=(
            "order_id",
            "nunique"
        )
    )
)


customer_360 = customer_360.merge(
    completed_order_counts,
    on="customer_id",
    how="left"
)


# ============================================================
# CANCELLED ORDERS
# ============================================================

cancelled_orders = orders[
    orders["order_status"]
    == "Cancelled"
]

cancelled_counts = (
    cancelled_orders
    .groupby(
        "customer_id",
        as_index=False
    )
    .agg(
        cancelled_orders=(
            "order_id",
            "nunique"
        )
    )
)


customer_360 = customer_360.merge(
    cancelled_counts,
    on="customer_id",
    how="left"
)


# ============================================================
# RETURNED ORDERS
# ============================================================

returned_orders = orders[
    orders["order_status"]
    == "Returned"
]

returned_counts = (
    returned_orders
    .groupby(
        "customer_id",
        as_index=False
    )
    .agg(
        returned_orders=(
            "order_id",
            "nunique"
        )
    )
)


customer_360 = customer_360.merge(
    returned_counts,
    on="customer_id",
    how="left"
)


# ============================================================
# ELIGIBLE ORDER ITEMS
# ============================================================

eligible_order_items = order_items.merge(
    completed_orders[
        [
            "order_id",
            "customer_id",
            "order_date"
        ]
    ],
    on="order_id",
    how="inner"
)


# ============================================================
# ITEM AND SPEND METRICS
# ============================================================

customer_spend = (
    eligible_order_items
    .groupby(
        "customer_id",
        as_index=False
    )
    .agg(
        total_items_purchased=(
            "quantity",
            "sum"
        ),
        total_spend=(
            "item_total",
            "sum"
        )
    )
)


customer_360 = customer_360.merge(
    customer_spend,
    on="customer_id",
    how="left"
)


# ============================================================
# FIRST ORDER DATE
# ============================================================

first_order = (
    completed_orders
    .groupby(
        "customer_id",
        as_index=False
    )
    .agg(
        first_order_date=(
            "order_date",
            "min"
        )
    )
)


customer_360 = customer_360.merge(
    first_order,
    on="customer_id",
    how="left"
)


# ============================================================
# LAST ORDER DATE
# ============================================================

last_order = (
    completed_orders
    .groupby(
        "customer_id",
        as_index=False
    )
    .agg(
        last_order_date=(
            "order_date",
            "max"
        )
    )
)


customer_360 = customer_360.merge(
    last_order,
    on="customer_id",
    how="left"
)


# ============================================================
# SUCCESSFUL PAYMENTS
# ============================================================

successful_payments = payments[
    payments["payment_status"]
    == "Success"
].copy()


successful_payment_amount = (
    successful_payments
    .groupby(
        "order_id",
        as_index=False
    )
    .agg(
        successful_payment_amount=(
            "amount",
            "sum"
        )
    )
)


# Connect successful payments to customers
payment_customer = successful_payment_amount.merge(
    orders[
        [
            "order_id",
            "customer_id"
        ]
    ],
    on="order_id",
    how="inner"
)


customer_payments = (
    payment_customer
    .groupby(
        "customer_id",
        as_index=False
    )
    .agg(
        successful_payment_amount=(
            "successful_payment_amount",
            "sum"
        )
    )
)


customer_360 = customer_360.merge(
    customer_payments,
    on="customer_id",
    how="left"
)


# ============================================================
# FILL MISSING AGGREGATIONS
# ============================================================

zero_columns = [
    "total_orders",
    "completed_orders",
    "cancelled_orders",
    "returned_orders",
    "total_items_purchased",
    "total_spend",
    "successful_payment_amount"
]

for column in zero_columns:

    customer_360[column] = (
        customer_360[column]
        .fillna(0)
    )


# ============================================================
# AVERAGE ORDER VALUE
# ============================================================

customer_360[
    "average_order_value"
] = 0.0

has_completed_orders = (
    customer_360[
        "completed_orders"
    ] > 0
)

customer_360.loc[
    has_completed_orders,
    "average_order_value"
] = (
    customer_360.loc[
        has_completed_orders,
        "total_spend"
    ]
    /
    customer_360.loc[
        has_completed_orders,
        "completed_orders"
    ]
)


# ============================================================
# CUSTOMER LIFETIME
# ============================================================

customer_360[
    "customer_lifetime_days"
] = pd.NA

has_orders = (
    customer_360[
        "first_order_date"
    ].notna()
    &
    customer_360[
        "last_order_date"
    ].notna()
)

customer_360.loc[
    has_orders,
    "customer_lifetime_days"
] = (
    customer_360.loc[
        has_orders,
        "last_order_date"
    ]
    -
    customer_360.loc[
        has_orders,
        "first_order_date"
    ]
).dt.days


# ============================================================
# ROUND FINANCIAL VALUES
# ============================================================

customer_360[
    "total_spend"
] = customer_360[
    "total_spend"
].round(2)

customer_360[
    "successful_payment_amount"
] = customer_360[
    "successful_payment_amount"
].round(2)

customer_360[
    "average_order_value"
] = customer_360[
    "average_order_value"
].round(2)


# ============================================================
# SORT
# ============================================================

customer_360 = customer_360.sort_values(
    "customer_id"
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
# WRITE GOLD
# ============================================================

customer_360.to_parquet(
    GOLD_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("CUSTOMER 360 GOLD DATASET")
print("========================================")

print(
    "Rows:",
    len(customer_360)
)

print(
    "Columns:",
    len(customer_360.columns)
)

print()
print("Columns:")

print(
    customer_360.columns.tolist()
)

print()
print("Preview:")

print(
    customer_360.head(10)
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("========================================")
print("FINAL CUSTOMER 360 VALIDATION")
print("========================================")

print(
    "Duplicate customer IDs:",
    customer_360[
        "customer_id"
    ].duplicated().sum()
)

print()
print("Null values:")

print(
    customer_360.isna().sum()
)

print()
print(
    "Negative total orders:",
    (
        customer_360[
            "total_orders"
        ] < 0
    ).sum()
)

print(
    "Negative total spend:",
    (
        customer_360[
            "total_spend"
        ] < 0
    ).sum()
)

print(
    "Negative successful payments:",
    (
        customer_360[
            "successful_payment_amount"
        ] < 0
    ).sum()
)

print()
print(
    "Customer 360 Gold dataset written successfully"
)

print(
    "Output:",
    GOLD_PATH
)