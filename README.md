# UrbanCart AWS Data Engineering

An end-to-end data engineering project simulating an e-commerce data platform for **UrbanCart**.

The project is being built as an interview/portfolio project to demonstrate practical data engineering concepts including data ingestion, data quality validation, data transformation, layered data architecture, Parquet-based processing, and eventually AWS cloud data engineering.

---

## 📌 Project Overview

UrbanCart is a fictional e-commerce platform where customers can browse products, place orders, make payments, and receive products through sellers and delivery partners.

The project simulates the complete data journey from raw source data to analytics-ready datasets.

The planned pipeline is:

```
Source Data
    ↓
Raw Layer
    ↓
Bronze Layer
    ↓
Data Quality Validation
    ↓
Silver Layer
    ↓
Gold Layer
    ↓
Analytics
```

The project will gradually move from local development to an AWS-based architecture.

---

## 🏗️ Current Architecture

At the current stage, the project is being developed locally using Python, Pandas, and Parquet.

```
                    UrbanCart Data
                          │
                          ▼
                    Raw Source Data
                          │
                          ▼
                       Bronze
                          │
                    Data Validation
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
           Silver                     DQ
       Trusted Data              Quarantine
              │
              ▼
        Future Gold Layer
```

### Data Layers

| Layer  | Purpose                                          |
|--------|---------------------------------------------------|
| Raw    | Original generated/source datasets                |
| Bronze | Raw data stored in a structured format             |
| Silver | Validated and trusted data                         |
| DQ     | Invalid/rejected records retained for investigation|
| Gold   | Future analytics/business-ready datasets           |

---

## 📊 Current Progress

### Customer Pipeline

The Customer dataset contains:

- Customer identification details
- Personal information
- Contact information
- Location information
- Registration information
- Loyalty information
- Customer status

**Customer Processing**

```
Customers Raw
     ↓
Bronze
     ↓
DQ Validation
     ↓
 ┌───┴────┐
 ▼        ▼
Silver    DQ
```

Current results:

| Layer  | Records |
|--------|---------|
| Bronze | 50,000  |
| Silver | 48,470  |
| DQ     | 1,530   |

Silver contains 14 business columns.
DQ contains the 14 business columns plus `_dq_issue`.

**Customer Data Quality Rules**

The pipeline currently validates:

- Missing customer ID
- Duplicate customer ID
- Missing first name
- Missing last name
- Invalid gender
- Invalid date of birth
- Invalid email
- Invalid phone
- Missing city
- Missing state
- Invalid country
- Invalid postal code
- Invalid registration date
- Invalid loyalty tier
- Invalid customer status

Invalid records are separated into the DQ layer rather than being deleted.

---

### 🏪 Seller Pipeline

The Seller dataset represents businesses/vendors selling products through UrbanCart.

Seller attributes include:

- Seller ID
- Seller name
- Email
- Phone
- City
- State
- Country
- Seller rating
- Seller status
- Joined date

**Seller Processing**

```
Sellers Raw
     ↓
Bronze
     ↓
DQ Validation
     ↓
 ┌───┴────┐
 ▼        ▼
Silver    DQ
```

Current results:

| Layer  | Records |
|--------|---------|
| Bronze | 1,000   |
| Silver | 958     |
| DQ     | 42      |

Silver contains 10 business columns.
DQ contains the 10 business columns plus `_dq_issue`.

**Seller Data Quality Rules**

The pipeline currently validates:

- Missing seller ID
- Duplicate seller ID
- Missing seller name
- Invalid email
- Invalid phone
- Missing city
- Missing state
- Invalid country
- Invalid seller rating
- Invalid seller status
- Invalid joined date
- Future joined date

---

## 📁 Project Structure

```
UrbanCart-AWS-Data-Engineering/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   │
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── customers.parquet
│   │   ├── sellers.csv
│   │   └── sellers.parquet
│   │
│   ├── bronze/
│   │   ├── customers/
│   │   │   ├── customers.csv
│   │   │   └── customers.parquet
│   │   │
│   │   └── sellers/
│   │       └── sellers.parquet
│   │
│   ├── silver/
│   │   ├── customers/
│   │   │   └── customers.parquet
│   │   │
│   │   └── sellers/
│   │       └── sellers.parquet
│   │
│   └── dq/
│       ├── customers/
│       │   └── customers_dq.parquet
│       │
│       └── sellers/
│           └── sellers_dq.parquet
│
└── scripts/
    ├── generate_urbancart_data.py
    ├── transform_customers.py
    └── transform_sellers.py
```

---

## 🛠️ Technologies

**Current**
- Python
- Pandas
- PyArrow
- Parquet
- Git
- GitHub

**Planned AWS Technologies**
- Amazon S3
- AWS Glue
- PySpark
- AWS IAM
- AWS Lambda
- AWS Step Functions
- Amazon Athena
- Amazon CloudWatch

Additional AWS services may be introduced as the project evolves.

---

## 🔍 Data Quality Approach

The project follows a simple quarantine-based data quality approach.

Instead of deleting invalid records:

```
Invalid Record
      ↓
DQ / Quarantine
      ↓
Investigation / Remediation
```

Valid records are promoted to Silver:

```
Valid Record
      ↓
Silver
      ↓
Analytics / Downstream Processing
```

This preserves the original data and provides traceability for rejected records.

---

## 🔗 Planned Data Model

The UrbanCart platform will eventually contain multiple related entities.

The planned relationships include:

```
Customer
   │
   ▼
Orders
   │
   ▼
Order Items
   │
   ├──────────► Products
   │                │
   │                ▼
   │              Sellers
   │
   └──────────► Payments
```

Additional entities such as warehouses, inventory, delivery partners, returns, reviews, and wishlists will be added as the project progresses.

---

## 🚀 Planned Roadmap

**Phase 1 — Data Foundation**
- [x] Customer dataset
- [x] Seller dataset
- [x] Raw layer
- [x] Bronze layer
- [x] Data quality validation
- [x] Silver layer
- [x] DQ / quarantine layer

**Phase 2 — E-commerce Data Model**
- [ ] Product dataset
- [ ] Order dataset
- [ ] Order Items dataset
- [ ] Payment dataset
- [ ] Warehouse dataset
- [ ] Inventory dataset
- [ ] Delivery dataset
- [ ] Returns dataset
- [ ] Reviews dataset

**Phase 3 — AWS Data Platform**
- [ ] Amazon S3 data lake
- [ ] AWS Glue
- [ ] PySpark transformations
- [ ] Glue Data Catalog
- [ ] Amazon Athena
- [ ] IAM security
- [ ] CloudWatch monitoring

**Phase 4 — Analytics Layer**
- [ ] Gold datasets
- [ ] Business KPIs
- [ ] Customer analytics
- [ ] Seller analytics
- [ ] Product analytics
- [ ] Sales analytics

**Phase 5 — Production Improvements**
- [ ] Incremental processing
- [ ] Partitioning
- [ ] Pipeline orchestration
- [ ] Error handling
- [ ] Logging
- [ ] Monitoring
- [ ] Data lineage
- [ ] CI/CD

---

## ▶️ Running the Project

Create and activate the Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Customer transformation:

```bash
python scripts/transform_customers.py
```

Run the Seller transformation:

```bash
python scripts/transform_sellers.py
```

---

## 🎯 Project Objective

The primary objective of this project is to build a practical end-to-end data engineering solution that demonstrates:

- Data ingestion
- Data validation
- Data quality management
- ETL/ELT processing
- Data lake architecture
- Dimensional/data modeling concepts
- PySpark processing
- AWS cloud services
- Pipeline orchestration
- Analytics-ready data preparation

The project is being developed incrementally, with each stage validated before moving to the next.

---

## 👨‍💻 Author

**Vrushabh Alase**
AWS | Data Engineering | Python | SQL | PySpark