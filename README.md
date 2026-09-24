# Zepto Data & AI Platform

This repository contains the capstone project for the Zepto Data & AI Platform.

## Project Modules

### Module 1 - Data Pipeline

The data pipeline:

- Scrapes book data from Books to Scrape.
- Cleans and transforms the scraped data.
- Converts GBP prices to INR using the fixed rate of 105.50 INR per GBP.
- Stores the data in a normalized SQLite database.
- Runs SQL queries for analysis.
- Reproduces a SQL JOIN using Pandas merge.

### Module 2 - Analytics

This folder will contain the analytics work for the project.

### Module 3 - Support Assistant

This folder will contain the support assistant work for the project.

## Repository Structure

    zepto-data-ai-platform/
    ├── README.md
    ├── data_pipeline/
    ├── analytics/
    └── support_assistant/

## Module 1 Results

The current Module 1 pipeline successfully scraped 67 books across 4 categories.

The SQL JOIN and equivalent Pandas merge both returned 67 rows and matched successfully.
