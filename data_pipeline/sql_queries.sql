-- Query 1: SELECT + WHERE

SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4


-- Query 2: ORDER BY

SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC


-- Query 3: ORDER BY + LIMIT

SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 5


-- Query 4: DISTINCT

SELECT DISTINCT category_name
FROM categories


-- Query 5: JOIN + BETWEEN

SELECT books.title, books.price_gbp, categories.category_name
FROM books
JOIN categories
ON books.category_id = categories.category_id
WHERE books.price_gbp BETWEEN 20 AND 30
ORDER BY books.price_gbp


