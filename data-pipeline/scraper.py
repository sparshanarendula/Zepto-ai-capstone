from urllib import response
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3

url_provided="https://books.toscrape.com/"

#here selecting the 3 categories as 1. Mystery, 2. Romance and 3. Non-fiction

#above are indexed as 3, 8 and 13 respectively in the website

categories = {
    "Mystery": "catalogue/category/books/mystery_3/index.html",
    "Romance": "catalogue/category/books/romance_8/index.html",
    "Non-fiction": "catalogue/category/books/nonfiction_13/index.html"
}

#creating an empty list to store book data

books_data = []

#looping through each category

for category_name, category_url in categories.items():

    #creating the complete URL

    url = url_provided + category_url

    #sending request to the website

    page = requests.get(url)

    #checking whether the website is accessible

    if page.status_code != 200:
        print(f"Could not access {category_name}")
        continue

    #creating BeautifulSoup object

    soup = BeautifulSoup(page.text, "html.parser")

    #selecting all book articles

    book_articles = soup.select("article.product_pod")

    #collecting first 20 books from each category

    for book in book_articles[:20]:

        #getting book title

        title = book.h3.a["title"]

        #getting book price

        price = book.select_one(".price_color").text.strip()

        #getting book rating

        rating_class = book.select_one("p.star-rating")["class"]
        rating = rating_class[1]

        #converting star rating into number

        rating_map = {
            "One": 1,
            "Two": 2,
            "Three": 3,
            "Four": 4,
            "Five": 5
        }
        rating = rating_map.get(rating)

        #getting book availability

        availability = book.select_one(".availability").text.strip()

        #adding the book details into the list

        books_data.append({
            "Title": title,
            "Price": price,
            "Rating": rating,
            "Availability": availability,
            "Category": category_name
        })

#creating DataFrame

df = pd.DataFrame(books_data)

#printing the scraped data

print("\nScraped Data:")
print(df.head(10))

#printing total number of books

print("\nTotal number of books:", len(df))

#cleaning availability column

df["Availability"] = df["Availability"].str.strip()

#creating books_cleaned DataFrame

books_cleaned = df.copy()

#cleaning Price column

books_cleaned["Price"] = books_cleaned["Price"].astype(str).str.strip()

#removing pound symbol and extracting numeric price

books_cleaned["Price"] = books_cleaned["Price"].str.replace("£", "", regex=False).replace("Â£", "", regex=False).replace("Ã‚Â£", "", regex=False).str.extract(r"(\d+(?:\.\d+)?)", expand=False)

#converting price into numeric value

books_cleaned["Price"] = pd.to_numeric(
    books_cleaned["Price"],
    errors="coerce"
)

#converting GBP to INR

gbp_to_inr = 127.00
books_cleaned["Price"] = books_cleaned["Price"] * gbp_to_inr

#rounding price to 2 decimal places

books_cleaned["Price"] = books_cleaned["Price"].round(2)

#converting rating into numeric value

books_cleaned["Rating"] = pd.to_numeric(
    books_cleaned["Rating"],
    errors="coerce"
).astype("Int64")

#removing duplicate books

books_cleaned = books_cleaned.drop_duplicates()

#resetting DataFrame index

books_cleaned = books_cleaned.reset_index(drop=True)

#printing cleaned DataFrame

print("\nCleaned Data with Price in INR:")
print(books_cleaned.head(10))

#checking for blank price values

print("\nBlank Price Values:")
print(books_cleaned["Price"].isna().sum())

#saving cleaned data to CSV

books_cleaned.to_csv("books_cleaned.csv", index=False)

#creating SQLite database connection

connection = sqlite3.connect("books_database.db")

#creating cursor

cursor = connection.cursor()

#creating categories table

cursor.execute(
    """CREATE TABLE IF NOT EXISTS categories
(
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
)
""")

#creating books table

cursor.execute(
    """CREATE TABLE IF NOT EXISTS books
(
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price REAL,
    rating INTEGER,
    availability TEXT,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
)
""")

#inserting categories into categories table

for category_name in books_cleaned["Category"].unique():
    cursor.execute(
        """INSERT OR IGNORE INTO categories (category_name)
        VALUES (?)""",
        (category_name,)
    )

#committing category changes

connection.commit()

#deleting existing books to avoid duplicate records when running the code again

cursor.execute("DELETE FROM books")

#resetting book_id sequence

cursor.execute("DELETE FROM sqlite_sequence WHERE name='books'")

#inserting book data into books table

for _, row in books_cleaned.iterrows():

    #finding category_id for the current category

    cursor.execute(
        """SELECT category_id FROM categories
        WHERE category_name = ?""",
        (row["Category"],)
    )
    category_id = cursor.fetchone()[0]

    #inserting book data into books table

    cursor.execute(
        """INSERT INTO books
        (title, price, rating, availability, category_id)
        VALUES (?, ?, ?, ?, ?)""",
        (
            row["Title"],
            row["Price"],
            row["Rating"],
            row["Availability"],
            category_id
        )
    )

#committing book changes

connection.commit()

#printing database insertion confirmation

print("\nData inserted successfully into SQLite database.")

#SQL Query 1 - SELECT

query_1 = """SELECT
book_id,
title,
price,
rating,
availability
FROM books"""

result_1 = pd.read_sql_query(query_1, connection)

print("\nSQL Query 1 - SELECT:")
print(result_1.head(10))

#SQL Query 2 - WHERE

query_2 = """SELECT
title,
price,
rating
FROM books
WHERE rating = 5"""

result_2 = pd.read_sql_query(query_2, connection)

print("\nSQL Query 2 - WHERE:")
print(result_2.head(10))

#SQL Query 3 - ORDER BY

query_3 = """SELECT
title,
price,
rating
FROM books
ORDER BY price DESC"""

result_3 = pd.read_sql_query(query_3, connection)

print("\nSQL Query 3 - ORDER BY:")
print(result_3.head(10))

#SQL Query 4 - LIMIT

query_4 = """SELECT
title,
price,
rating
FROM books
ORDER BY rating DESC
LIMIT 10"""

result_4 = pd.read_sql_query(query_4, connection)

print("\nSQL Query 4 - LIMIT:")
print(result_4)

#SQL Query 5 - DISTINCT

query_5 = """SELECT DISTINCT
category_id
FROM books"""

result_5 = pd.read_sql_query(query_5, connection)

print("\nSQL Query 5 - DISTINCT:")
print(result_5)

#SQL Query 6 - JOIN

query_6 = """SELECT
b.book_id,
b.title,
b.price,
b.rating,
b.availability,
c.category_name
FROM books b
JOIN categories c
ON b.category_id = c.category_id"""

result_6 = pd.read_sql_query(query_6, connection)

print("\nSQL Query 6 - JOIN:")
print(result_6.head(10))

#SQL Query 7 - JOIN + WHERE + ORDER BY + LIMIT

query_7 = """SELECT
b.title,
b.price,
b.rating,
c.category_name
FROM books b
JOIN categories c
ON b.category_id = c.category_id
WHERE b.rating = 5
ORDER BY b.price DESC
LIMIT 10"""

result_7 = pd.read_sql_query(query_7, connection)

print("\nSQL Query 7 - JOIN + WHERE + ORDER BY + LIMIT:")
print(result_7)

#SQL Query 8 - filtering Mystery category

query_8 = """SELECT
b.title,
b.price,
b.rating,
c.category_name
FROM books b
JOIN categories c
ON b.category_id = c.category_id
WHERE c.category_name = 'Mystery'
ORDER BY b.price DESC"""

result_8 = pd.read_sql_query(query_8, connection)

print("\nSQL Query 8 - Mystery Category:")
print(result_8)

#reading books table into Pandas

books_df = pd.read_sql_query(
    """SELECT * FROM books""",
    connection
)

#reading categories table into Pandas

categories_df = pd.read_sql_query(
    """SELECT * FROM categories""",
    connection
)

#merging books and categories using Pandas

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

#printing Pandas merged DataFrame

print("\nPandas merged DataFrame:")
print(merged_df.head(10))

#selecting required columns

final_df = merged_df[
    [
        "book_id",
        "title",
        "price",
        "rating",
        "availability",
        "category_name"
    ]
]

#printing final DataFrame

print("\nFinal DataFrame with selected columns:")
print(final_df.head(10))

#comparing SQL JOIN result with Pandas merge result

sql_join_compare = result_6[
    [
        "book_id",
        "title",
        "price",
        "rating",
        "availability",
        "category_name"
    ]
].sort_values(
    by="book_id"
).reset_index(drop=True)

pandas_join_result = merged_df[
    [
        "book_id",
        "title",
        "price",
        "rating",
        "availability",
        "category_name"
    ]
]

pandas_join_compare = pandas_join_result.sort_values(
    by="book_id"
).reset_index(drop=True)

#checking whether SQL JOIN and Pandas merge produced the same result

if sql_join_compare.equals(pandas_join_compare):
    print("\nSuccess: SQL JOIN and Pandas merge produced the same result.")
else:
    print("\nThe SQL JOIN and Pandas merge results are different.")

#closing database connection

connection.close()

#printing final checklist

print("\nProject completed successfully.")
print(f"Total books scraped: {len(books_cleaned)}")
print("Categories scraped: Mystery, Romance, Non-fiction")
print("Price converted from GBP to INR.")
print("Data cleaned and stored in SQLite database.")
print("SQL queries executed successfully.")
print("SQL JOIN and Pandas merge comparison completed.")