
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3


def scrape_category(category_url, category_name):
    books = []

    response = requests.get(category_url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for book in soup.select("article.product_pod"):
        title = book.h3.a["title"]
        price = book.select_one(".price_color").get_text(strip=True)
        star_rating = book.select_one(".star-rating")["class"][1]
        availability = book.select_one(".availability").get_text(" ", strip=True)

        books.append({
            "title": title,
            "price": price,
            "star_rating": star_rating,
            "availability": availability,
            "category": category_name
        })

    return books


# Category pages to scrape

categories = {
    "Travel": "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",
    "Mystery": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "Science Fiction": "https://books.toscrape.com/catalogue/category/books/science-fiction_16/index.html",
    "Historical Fiction": "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html"
}


# Scrape all categories

all_books = []

for category_name, category_url in categories.items():
    books = scrape_category(category_url, category_name)
    all_books.extend(books)

print("Total books scraped:", len(all_books))


# Convert scraped books into a DataFrame

df = pd.DataFrame(all_books)

print("Rows:", len(df))
print("Columns:", len(df.columns))

# Clean the price and convert it to GBP

df["price_gbp"] = pd.to_numeric(
    df["price"].str.extract(r"([\d.]+)", expand=False),
    errors="coerce"
)

# Handle price parsing failures using median imputation

if df["price_gbp"].isna().any():
    median_price = df["price_gbp"].median()
    df["price_gbp"] = df["price_gbp"].fillna(median_price)


# Convert star ratings from text to numbers

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(rating_map)

# Handle rating parsing failures

if df["rating"].isna().any():
    median_rating = round(df["rating"].median())
    df["rating"] = df["rating"].fillna(median_rating)

df["rating"] = df["rating"].astype(int)


# Convert availability text to a boolean

df["in_stock"] = df["availability"].str.contains(
    "In stock",
    case=False,
    na=False
)


# Convert GBP to INR using the fixed project exchange rate

GBP_TO_INR = 105.50

df["price_inr"] = df["price_gbp"] * GBP_TO_INR


# Validation

print("Missing price values:", df["price_gbp"].isna().sum())
print("Price data type:", df["price_gbp"].dtype)

print("Missing rating values:", df["rating"].isna().sum())
print("Rating data type:", df["rating"].dtype)

print("Missing in_stock values:", df["in_stock"].isna().sum())
print("in_stock data type:", df["in_stock"].dtype)

print("Missing price_inr values:", df["price_inr"].isna().sum())
print("price_inr data type:", df["price_inr"].dtype)

# Create SQLite database

conn = sqlite3.connect("books.db")
cursor = conn.cursor()

# Enable foreign key support
cursor.execute("PRAGMA foreign_keys = ON")


# Recreate tables each time the pipeline runs
# This prevents duplicate records when the script is rerun

cursor.execute("DROP TABLE IF EXISTS books")
cursor.execute("DROP TABLE IF EXISTS categories")


# Categories table

cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
)
""")


# Books table

cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock BOOLEAN,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
)
""")

conn.commit()

print("Database and tables created successfully!")

# Insert categories into the categories table

for category_name in df["category"].unique():
    cursor.execute(
        """
        INSERT INTO categories (category_name)
        VALUES (?)
        """,
        (category_name,)
    )

conn.commit()

print("Categories inserted successfully!")


# Insert books into the books table

for _, row in df.iterrows():

    category_id = cursor.execute(
        """
        SELECT category_id
        FROM categories
        WHERE category_name = ?
        """,
        (row["category"],)
    ).fetchone()[0]

    cursor.execute(
        """
        INSERT INTO books
        (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            row["in_stock"],
            category_id
        )
    )

conn.commit()


# Verify the number of books in the database

cursor.execute("SELECT COUNT(*) FROM books")
book_count = cursor.fetchone()[0]

print("Books in database:", book_count)


# Close the database connection

conn.close()
