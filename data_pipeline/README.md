# Module 1 - Data Pipeline

## Overview

This module scrapes book data from Books to Scrape, cleans the data, converts GBP prices to INR, stores the data in a normalized SQLite database, and runs SQL and Pandas analysis.

## Data Source

Website: Books to Scrape

The scraper collects books from these categories:

* Travel
* Mystery
* Science Fiction
* Historical Fiction

The final dataset contains 67 books across 4 categories.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## How to Run

First run the scraper and database pipeline:

```bash
python scrape_and_load.py
```

This will:

1. Scrape the selected book categories.
2. Clean the price, rating, and availability fields.
3. Calculate `price_inr` using the required fixed conversion rate.
4. Create the SQLite database.
5. Create the `categories` and `books` tables.
6. Insert the cleaned data into the database.

Then run the SQL queries:

```bash
python run_queries.py
```

The SQL query strings are stored in `sql_queries.sql`.

The five query outputs are saved as CSV files:

* `query1_output.csv`
* `query2_output.csv`
* `query3_output.csv`
* `query4_output.csv`
* `query5_output.csv`

## Data Cleaning

### Price

The original price is scraped as text in GBP.

The numeric GBP value is extracted and converted to a float column named `price_gbp`.

If a price cannot be parsed, the numeric median price is used for imputation.

### Rating

The website provides ratings as text such as `One`, `Two`, `Three`, `Four`, and `Five`.

These values are converted to integers from 1 to 5 in the `rating` column.

If a rating cannot be parsed, the rounded median rating is used for imputation.

### Availability

The availability text is converted into a Boolean field called `in_stock`.

If the availability text contains `In stock`, the value is `True`; otherwise, it is `False`.

### Price Conversion

The project uses the required fixed baseline conversion rate:

**1 GBP = 105.50 INR**

The INR price is calculated as:

```text
price_inr = price_gbp * 105.50
```

This is a project-defined fixed conversion rate. No external currency API is required.

## Database Schema

The SQLite database is stored in `books.db`.

It contains two normalized tables with a primary-key/foreign-key relationship.

### categories

| Column        | Type    | Description          |
| ------------- | ------- | -------------------- |
| category_id   | INTEGER | Primary key          |
| category_name | TEXT    | Unique category name |

### books

| Column      | Type    | Description                        |
| ----------- | ------- | ---------------------------------- |
| book_id     | INTEGER | Primary key                        |
| title       | TEXT    | Book title                         |
| price_gbp   | REAL    | Cleaned price in GBP               |
| price_inr   | REAL    | Converted price in INR             |
| rating      | INTEGER | Rating from 1 to 5                 |
| in_stock    | BOOLEAN | Whether the book is in stock       |
| category_id | INTEGER | Foreign key referencing categories |

The `books.category_id` column references `categories.category_id`.

## SQL Queries

Five SQL queries are included in `sql_queries.sql`.

### Query 1

Find books with a rating of 4 or higher.

Demonstrates:

* SELECT
* WHERE

### Query 2

List books ordered from highest to lowest GBP price.

Demonstrates:

* ORDER BY

### Query 3

Find the five most expensive books.

Demonstrates:

* ORDER BY
* LIMIT

### Query 4

List the unique book categories.

Demonstrates:

* DISTINCT

### Query 5

Find books priced between £20 and £30 and display their categories.

Demonstrates:

* JOIN
* WHERE
* BETWEEN
* ORDER BY

The query outputs are stored as CSV files in this folder.

## Pandas Analysis

At least two SQL query results are read back into Pandas DataFrames using `pd.read_sql()`.

The SQL JOIN result is also reproduced directly in memory using `pd.merge()` between the books and categories DataFrames.

Both approaches produce equivalent results.

Example validation:

```text
Do SQL JOIN and Pandas merge match? True
```

The SQL JOIN and Pandas merge both contain 67 matching book-category rows.

## Design Decisions

* The fixed conversion rate of **1 GBP = 105.50 INR** is used as required by the project.
* Numeric parsing failures are handled using median imputation.
* Invalid or unexpected availability text is converted to `False`.
* A normalized two-table SQLite schema is used to avoid repeating category names in every book record.
* The pipeline can be regenerated from scratch by running `scrape_and_load.py`.
* SQL query outputs are saved as CSV files for inspection and reproducibility.
