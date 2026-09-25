# Zepto Data & AI Platform

This repository contains the capstone project for the Zepto Data & AI Platform.

The project combines data collection and processing, analytics and machine learning, and a retrieval-based customer support assistant into a single repository.

## Project Modules

### Module 1 - Data Pipeline

The data pipeline:

* Scrapes book data from Books to Scrape.
* Cleans and transforms the scraped data.
* Converts GBP prices to INR using the fixed rate of 105.50 INR per GBP.
* Stores the data in a normalized SQLite database.
* Runs SQL queries for analysis.
* Reproduces a SQL JOIN using Pandas merge.

**Module location:** `data_pipeline/`

### Module 2 - Analytics

The analytics module performs exploratory data analysis and machine learning using the Titanic dataset.

It includes:

* Data loading and cleaning.
* Missing-value analysis and treatment.
* Exploratory data analysis and visualizations.
* Correlation analysis.
* Feature standardization.
* Classification models.
* Imbalance handling using class weights and SMOTE.
* Hyperparameter tuning.
* Regression analysis.
* Model evaluation and persistence.

**Module location:** `analytics/`

### Module 3 - Support Assistant

The support assistant implements a customer-support question-answering workflow using:

* LangGraph for workflow orchestration.
* Sentence Transformers for semantic embeddings.
* ChromaDB for vector retrieval.
* Pydantic for structured response validation.
* FastAPI for the `/ask` API.
* Docker for containerized execution.

The required baseline runs deterministically with `MOCK_LLM=1`.

**Module location:** `support_assistant/`

## Repository Structure

```text
zepto-data-ai-platform/
├── README.md
├── .gitignore
├── data_pipeline/
├── analytics/
└── support_assistant/
```

Each module contains its own code, data, documentation, and supporting files as required for that module.

## Module 1 Results

The Module 1 pipeline successfully scraped **67 books across 4 categories**.

The SQL JOIN and equivalent Pandas merge both returned **67 rows** and matched successfully.

## Module 2 Results

The Titanic analytics module includes classification and regression experiments.

The classification workflow evaluates multiple models using accuracy, precision, recall, F1 score, and ROC-AUC. It also includes imbalance handling, hyperparameter tuning, and model persistence.

The final model-selection decision and detailed evaluation results are documented in:

`analytics/README.md`

## Module 3 Results

The Support Assistant provides a FastAPI `/ask` endpoint backed by a LangGraph workflow.

Policy questions are routed through semantic retrieval using Sentence Transformers and ChromaDB, while general questions are handled through the direct-answer path.

The baseline application runs with:

```text
MOCK_LLM=1
```

Detailed architecture, API examples, and module documentation are available in:

`support_assistant/README.md`
