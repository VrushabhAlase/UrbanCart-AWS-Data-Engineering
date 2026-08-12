"""
UrbanCart Synthetic Data Generator
-----------------------------------
Generates realistic, intentionally "dirty" retail data for the
AWS Data Engineering interview-prep project (Bronze/Silver/Gold pipeline).

Usage:
    python generate_urbancart_data.py --outdir ./data --days 90 --seed 42

Outputs (mirrors the S3 raw/ layout from Step 0 design doc):
    data/customers/ingest_date=YYYY-MM-DD/customers.csv
    data/products/ingest_date=YYYY-MM-DD/products.csv
    data/stores/ingest_date=YYYY-MM-DD/stores.csv
    data/orders/order_date=YYYY-MM-DD/orders_batchN.json
    data/order_items/order_date=YYYY-MM-DD/order_items_batchN.json

All "dirty data" injection rates are configurable via the CONFIG dict below,
so you can regenerate cleaner or dirtier data to practice different scenarios.
"""

import argparse
import json
import os
import random
import uuid
from datetime import datetime, timedelta

from faker import Faker

fake = Faker("en_IN")

# ----------------------------------------------------------------------
# CONFIG — tune these to make data cleaner/dirtier for different practice runs
# ----------------------------------------------------------------------
CONFIG = {
    "num_customers": 50_000,
    "num_products": 5_000,
    "num_stores": 200,
    "avg_orders_per_day": 4_000,
    "avg_items_per_order": 2.3,

    # dirty-data injection rates (0.0 - 1.0)
    "customer_dup_rate": 0.02,          # duplicate customer_id rows
    "customer_null_email_rate": 0.03,
    "customer_bad_phone_rate": 0.02,
    "product_null_category_rate": 0.015,
    "product_price_as_string_rate": 0.01,   # price stored as "499.00" text glitch e.g. "₹499"
    "order_dup_rate": 0.01,             # duplicate order_id across daily files
    "order_orphan_customer_rate": 0.02, # order references non-existent customer_id
    "order_item_null_qty_rate": 0.015,
    "order_item_bad_total_rate": 0.02,  # line_total != qty * unit_price
    "order_item_negative_qty_rate": 0.005,  # simulated returns
    "schema_change_after_day": 45,      # discount_code field appears from this day onward
    "late_arrival_rate": 0.01,          # small % of a day's orders get order_date shifted -2 days
}

REGIONS = ["North", "South", "East", "West", "Central"]
STATE_TO_REGION = {
    "Maharashtra": "West", "Gujarat": "West", "Rajasthan": "North",
    "Delhi": "North", "Punjab": "North", "Karnataka": "South",
    "Tamil Nadu": "South", "Kerala": "South", "Telangana": "South",
    "West Bengal": "East", "Odisha": "East", "Bihar": "East",
    "Madhya Pradesh": "Central", "Uttar Pradesh": "Central",
}
STATES = list(STATE_TO_REGION.keys())
CATEGORIES = {
    "Electronics": ["Mobiles", "Laptops", "Accessories", "Audio"],
    "Fashion": ["Men", "Women", "Kids", "Footwear"],
    "Home": ["Furniture", "Kitchen", "Decor"],
    "Grocery": ["Staples", "Snacks", "Beverages"],
    "Beauty": ["Skincare", "Haircare", "Makeup"],
}
PAYMENT_MODES = ["UPI", "CREDIT_CARD", "DEBIT_CARD", "COD", "NET_BANKING"]
CHANNELS = ["ONLINE", "MOBILE_APP", "STORE"]
ORDER_STATUSES = ["COMPLETED", "CANCELLED", "RETURNED", "PENDING"]


def generate_customers(n, cfg):
    rows = []
    for i in range(n):
        cid = f"CUST-{i:07d}"
        state = random.choice(STATES)
        email = fake.email()
        if random.random() < cfg["customer_null_email_rate"]:
            email = ""
        phone = fake.phone_number()
        if random.random() < cfg["customer_bad_phone_rate"]:
            phone = "N/A"
        row = {
            "customer_id": cid,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "email": email,
            "phone": phone,
            "city": fake.city(),
            "state": state,
            "signup_date": fake.date_between(start_date="-3y", end_date="-1d").isoformat(),
            "customer_segment": random.choice(["Regular", "Premium", "VIP"]),
        }
        rows.append(row)
        # inject duplicate rows (same customer_id re-sent, simulating source resend)
        if random.random() < cfg["customer_dup_rate"]:
            rows.append(row.copy())
    random.shuffle(rows)
    return rows


def generate_products(n, cfg):
    rows = []
    for i in range(n):
        pid = f"PROD-{i:06d}"
        category = random.choice(list(CATEGORIES.keys()))
        sub_category = random.choice(CATEGORIES[category])
        if random.random() < cfg["product_null_category_rate"]:
            category = ""
        price = round(random.uniform(99, 49999), 2)
        price_field = price
        if random.random() < cfg["product_price_as_string_rate"]:
            price_field = f"Rs.{price}"  # dirty: price stored as text w/ currency symbol
        rows.append({
            "product_id": pid,
            "product_name": f"{fake.word().capitalize()} {sub_category} {random.choice(['Pro','Lite','Plus','Max',''])}".strip(),
            "category": category,
            "sub_category": sub_category,
            "brand": fake.company(),
            "unit_price": price_field,
            "is_active": random.choice([True, True, True, False]),
        })
    return rows


def generate_stores(n):
    rows = []
    for i in range(n):
        state = random.choice(STATES)
        rows.append({
            "store_id": f"STR-{i:04d}",
            "store_name": f"UrbanCart {fake.city()} Store",
            "city": fake.city(),
            "state": state,
            "region": STATE_TO_REGION[state],
        })
    return rows


def generate_orders_and_items(customer_ids, product_ids, store_ids, start_date, days, cfg):
    """Yields (order_date_str, orders_list, order_items_list) per day."""
    used_order_ids = set()
    for d in range(days):
        cur_date = start_date + timedelta(days=d)
        cur_date_str = cur_date.isoformat()
        n_orders = max(1, int(random.gauss(cfg["avg_orders_per_day"], cfg["avg_orders_per_day"] * 0.15)))

        daily_orders = []
        daily_items = []
        include_discount_code = d >= cfg["schema_change_after_day"]

        for _ in range(n_orders):
            order_id = f"ORD-{uuid.uuid4().hex[:10].upper()}"
            used_order_ids.add(order_id)

            # orphan customer injection
            if random.random() < cfg["order_orphan_customer_rate"]:
                customer_id = f"CUST-{random.randint(900000, 999999):07d}"  # doesn't exist in customers table
            else:
                customer_id = random.choice(customer_ids)

            # late arrival: order_date field says one thing, but it's placed in an earlier day's "file"
            # (simulated by just tagging it - the file partition is still cur_date, representing a late-landing record)
            effective_order_date = cur_date_str
            if random.random() < cfg["late_arrival_rate"] and d >= 2:
                effective_order_date = (cur_date - timedelta(days=2)).isoformat()

            order = {
                "order_id": order_id,
                "customer_id": customer_id,
                "store_id": random.choice(store_ids),
                "order_date": effective_order_date,
                "order_status": random.choice(ORDER_STATUSES),
                "payment_mode": random.choice(PAYMENT_MODES),
                "channel": random.choice(CHANNELS),
            }
            if include_discount_code:
                # schema evolution: new field appears from day 45 onward
                if random.random() < 0.3:
                    order["discount_code"] = random.choice(["SAVE10", "FESTIVE20", "WELCOME5", None])

            daily_orders.append(order)

            # duplicate order injection (order appears twice in the same/different batch)
            if random.random() < cfg["order_dup_rate"]:
                daily_orders.append(order.copy())

            # order items for this order
            n_items = max(1, int(random.gauss(cfg["avg_items_per_order"], 1)))
            for _ in range(n_items):
                product_id = random.choice(product_ids)
                qty = random.randint(1, 5)
                unit_price = round(random.uniform(99, 9999), 2)
                line_total = round(qty * unit_price, 2)

                if random.random() < cfg["order_item_null_qty_rate"]:
                    qty = None
                if random.random() < cfg["order_item_negative_qty_rate"]:
                    qty = -abs(random.randint(1, 3))  # return
                if random.random() < cfg["order_item_bad_total_rate"]:
                    line_total = round(line_total * random.uniform(1.2, 2.0), 2)  # mismatch injected

                daily_items.append({
                    "order_id": order_id,
                    "product_id": product_id,
                    "quantity": qty,
                    "unit_price": unit_price,
                    "line_total": line_total,
                })

        yield cur_date_str, daily_orders, daily_items


def write_csv(rows, path):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_json_lines(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./data")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default="2026-05-13")  # 90 days before ~Aug 10
    args = parser.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)

    cfg = CONFIG
    today = datetime.fromisoformat(args.start_date).date()

    print("Generating customers...")
    customers = generate_customers(cfg["num_customers"], cfg)
    ingest_date = datetime.now().date().isoformat()
    write_csv(customers, f"{args.outdir}/customers/ingest_date={ingest_date}/customers.csv")
    print(f"  -> {len(customers)} rows (includes injected duplicates)")

    print("Generating products...")
    products = generate_products(cfg["num_products"], cfg)
    write_csv(products, f"{args.outdir}/products/ingest_date={ingest_date}/products.csv")
    print(f"  -> {len(products)} rows")

    print("Generating stores...")
    stores = generate_stores(cfg["num_stores"])
    write_csv(stores, f"{args.outdir}/stores/ingest_date={ingest_date}/stores.csv")
    print(f"  -> {len(stores)} rows")

    customer_ids = [c["customer_id"] for c in customers]
    product_ids = [p["product_id"] for p in products]
    store_ids = [s["store_id"] for s in stores]

    print(f"Generating {args.days} days of orders + order_items...")
    total_orders, total_items = 0, 0
    for date_str, orders, items in generate_orders_and_items(
        customer_ids, product_ids, store_ids, today, args.days, cfg
    ):
        write_json_lines(orders, f"{args.outdir}/orders/order_date={date_str}/orders_batch1.json")
        write_json_lines(items, f"{args.outdir}/order_items/order_date={date_str}/order_items_batch1.json")
        total_orders += len(orders)
        total_items += len(items)

    print(f"  -> {total_orders} order records, {total_items} order_item records")
    print("\nDone. Data written to:", os.path.abspath(args.outdir))
    print(f"Total records: {len(customers) + len(products) + len(stores) + total_orders + total_items:,}")


if __name__ == "__main__":
    main()
