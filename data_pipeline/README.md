# Module 1 - Data Pipeline

## Overview

This module scrapes book data from Books to Scrape, cleans the data, converts GBP prices to INR, stores the data in a normalized SQLite database, and runs SQL and Pandas analysis.

## Data Source

Website: Books to Scrape

The scraper collects books from these categories:

- Travel
- Mystery
- Science Fiction
- Historical Fiction

The final dataset contains 67 books.

## Requirements

Install the required Python packages using:

    pip install -r requirements.txt

## How to Run

First run the scraper and database pipeline:

    python scrape_and_load.py

This will:

1. Scrape the selected book categories.
2. Clean the price, rating, and availability fields.
3. Calculate price_inr.
4. Create the SQLite database.
5. Create the categories and books tables.
6. Insert the scraped data into the database.

Then run the SQL queries:

    python run_queries.py

The SQL query strings are stored in sql_queries.sql.

The five query outputs are saved as CSV files:

- query1_output.csv
- query2_output.csv
- query3_output.csv
- query4_output.csv
- query5_output.csv

## Data Cleaning

### Price

The original price is scraped as text.

The numeric GBP value is extracted and converted to a float using pandas.

If a price cannot be parsed, the numeric median price is used for imputation.

### Rating

The website provides ratings as text such as One, Two, Three, Four, and Five.

These values are converted to integers from 1 to 5.

If a rating cannot be parsed, the rounded median rating is used for imputation.

### Availability

The availability text is converted into a Boolean field called `in_stock`.

If the availability text contains "In stock", the value is True. Otherwise, it is False.

### Price Conversion

The project uses the fixed conversion rate:

1 GBP = 105.50 INR

The INR price is calculated as:

    price_inr = price_gbp * 105.50

## Database Schema

The SQLite database is stored in `books.db`.

It contains two normalized tables:

### categories

| Column | Type | Description |
|---|---|---|
| category_id | INTEGER | Primary key |
| category_name | TEXT | Unique category name |

### books

| Column | Type | Description |
|---|---|---|
| book_id | INTEGER | Primary key |
| title | TEXT | Book title |
| price_gbp | REAL | Cleaned price in GBP |
| price_inr | REAL | Converted price in INR |
| rating | INTEGER | Rating from 1 to 5 |
| in_stock | BOOLEAN | Whether the book is in stock |
| category_id | INTEGER | Foreign key referencing categories |

The `books.category_id` column references `categories.category_id`.

## SQL Queries

Five SQL queries are included in `sql_queries.sql`.

### Query 1

Find books with a rating of 4 or higher.

Demonstrates:

- SELECT
- WHERE

### Query 2

List books ordered from highest to lowest GBP price.

Demonstrates:

- ORDER BY

### Query 3

Find the five most expensive books.

Demonstrates:

- ORDER BY
- LIMIT

### Query 4

List the unique book categories.

Demonstrates:

- DISTINCT

### Query 5

Find books priced between £20 and £30 and display their categories.

Demonstrates:

- JOIN
- WHERE
- BETWEEN
- ORDER BY

The query outputs are stored as CSV files in this folder.

## Pandas JOIN Comparison

The SQL JOIN between `books` and `categories` is reproduced using Pandas `merge()`.

Both results contain 67 rows.

The SQL JOIN and Pandas merge produce matching results:

    Do SQL JOIN and Pandas merge match? True

This demonstrates that the relational SQL JOIN and the equivalent in-memory Pandas merge produce the same result.
