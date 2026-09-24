
import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("books.db")

print("Connected to books.db")

# Query 1: SELECT + WHERE
query1 = """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
"""

# Query 2: ORDER BY
query2 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
"""

# Query 3: ORDER BY + LIMIT
query3 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 5
"""

# Query 4: DISTINCT
query4 = """
SELECT DISTINCT category_name
FROM categories
"""

# Query 5: JOIN + BETWEEN + WHERE + ORDER BY
query5 = """
SELECT books.title, books.price_gbp, categories.category_name
FROM books
JOIN categories
ON books.category_id = categories.category_id
WHERE books.price_gbp BETWEEN 20 AND 30
ORDER BY books.price_gbp
"""

print("All 5 SQL queries defined successfully!")

# Run all 5 queries and save their outputs

result1 = pd.read_sql(query1, conn)
result2 = pd.read_sql(query2, conn)
result3 = pd.read_sql(query3, conn)
result4 = pd.read_sql(query4, conn)
result5 = pd.read_sql(query5, conn)

result1.to_csv("query1_output.csv", index=False)
result2.to_csv("query2_output.csv", index=False)
result3.to_csv("query3_output.csv", index=False)
result4.to_csv("query4_output.csv", index=False)
result5.to_csv("query5_output.csv", index=False)

print("All 5 query outputs saved successfully!")
print("Query 1 rows:", len(result1))
print("Query 2 rows:", len(result2))
print("Query 3 rows:", len(result3))
print("Query 4 rows:", len(result4))
print("Query 5 rows:", len(result5))

# Save all SQL query strings to a .sql file

with open("sql_queries.sql", "w") as f:
    f.write("-- Query 1: SELECT + WHERE\n")
    f.write(query1 + "\n\n")

    f.write("-- Query 2: ORDER BY\n")
    f.write(query2 + "\n\n")

    f.write("-- Query 3: ORDER BY + LIMIT\n")
    f.write(query3 + "\n\n")

    f.write("-- Query 4: DISTINCT\n")
    f.write(query4 + "\n\n")

    f.write("-- Query 5: JOIN + BETWEEN\n")
    f.write(query5 + "\n\n")

print("SQL query strings saved to sql_queries.sql")

# Reproduce the SQL JOIN using Pandas merge

books_df = pd.read_sql("""
SELECT title, price_gbp, price_inr, rating, in_stock, category_id
FROM books
""", conn)

categories_df = pd.read_sql("""
SELECT category_id, category_name
FROM categories
""", conn)

pandas_join = books_df.merge(
    categories_df,
    on="category_id",
    how="inner"
)

pandas_join = pandas_join[
    ["title", "price_gbp", "category_name"]
]

# SQL JOIN result for comparison
sql_join = pd.read_sql("""
SELECT
    books.title,
    books.price_gbp,
    categories.category_name
FROM books
JOIN categories
ON books.category_id = categories.category_id
""", conn)

# Sort both results before comparing
sql_join = sql_join.sort_values("title").reset_index(drop=True)
pandas_join = pandas_join.sort_values("title").reset_index(drop=True)

print("SQL JOIN rows:", len(sql_join))
print("Pandas merge rows:", len(pandas_join))
print("Do SQL JOIN and Pandas merge match?",
      sql_join.equals(pandas_join))

print("\nComparison:")
comparison = pd.concat(
    [
        sql_join.add_prefix("SQL_"),
        pandas_join.add_prefix("Pandas_")
    ],
    axis=1
)

print(comparison.head(10))

# Close the database connection
conn.close()

print("\nDatabase connection closed.")
print("run_queries.py completed successfully!")
