# Zepto Data & AI Platform

This repository contains the capstone project for the Zepto Data & AI Platform.

The project combines data collection and processing, analytics and machine learning, and a retrieval-based customer support assistant into a single repository.

## Project Setup

This project uses **a separate `requirements.txt` file for each module**.

Each module should be installed and run from its own directory so that its dependencies remain isolated.

---

## Module 1 - Data Pipeline

### Overview

The data pipeline:

* Scrapes book data from Books to Scrape.
* Cleans and transforms the scraped data.
* Converts GBP prices to INR using the fixed rate of 105.50 INR per GBP.
* Stores the data in a normalized SQLite database.
* Runs SQL queries for analysis.
* Reproduces a SQL JOIN using Pandas merge.

### Location

```text
data_pipeline/
```

### Setup

Open a terminal in the `data_pipeline` directory:

```bash
cd data_pipeline
pip install -r requirements.txt
```

### Run

Run the scraper and database pipeline:

```bash
python scrape_and_load.py
```

Then run the SQL and Pandas analysis:

```bash
python run_queries.py
```

### Design Decisions

The pipeline separates scraping, cleaning, database loading, and analysis. A normalized SQLite structure is used to avoid unnecessary duplication, while a fixed GBP-to-INR conversion rate provides reproducible results. SQL analysis is validated against an equivalent Pandas merge.

### Results

The Module 1 pipeline successfully scraped **67 books across 4 categories**.

The SQL JOIN and equivalent Pandas merge both returned **67 rows** and matched successfully.

Detailed documentation is available in:

```text
data_pipeline/README.md
```

---

## Module 2 - Analytics

### Overview

The analytics module performs exploratory data analysis and machine learning using the Titanic dataset.

It includes:

* Data loading and cleaning.
* Missing-value analysis and treatment.
* Exploratory data analysis and visualizations.
* Correlation analysis.
* Feature standardization.
* Classification models.
* Class-imbalance handling using class weights and SMOTE.
* Hyperparameter tuning.
* Regression analysis.
* Model evaluation.
* Model persistence.

### Location

```text
analytics/
```

### Setup

Open a terminal in the `analytics` directory:

```bash
cd analytics
pip install -r requirements.txt
```

### Run

Run the exploratory data analysis:

```bash
python 01_eda.py
```

Then run the modeling workflow:

```bash
python 02_modeling.py
```

### Design Decisions

The module uses a reproducible preprocessing and modeling workflow with fixed random seeds and a stratified train/test split. Missing values are handled according to their missingness level, preprocessing is placed inside pipelines to reduce data leakage, and class imbalance is evaluated using both class weighting and SMOTE. Multiple classification and regression metrics are used to compare model behavior.

### Results

The module evaluates Logistic Regression, Decision Tree, Random Forest, Balanced Logistic Regression, SMOTE Logistic Regression, and a tuned Random Forest.

The detailed evaluation results, model comparison, and saved-model information are documented in:

```text
analytics/README.md
```

---

## Module 3 - Support Assistant

### Overview

The support assistant implements a customer-support question-answering workflow using:

* LangGraph for workflow orchestration.
* Sentence Transformers for semantic embeddings.
* `all-MiniLM-L6-v2` for embeddings.
* ChromaDB for vector storage and retrieval.
* Pydantic for structured response validation.
* FastAPI for the `/ask` API.
* Docker for containerized execution.

The required baseline runs deterministically with:

```text
MOCK_LLM=1
```

### Location

```text
support_assistant/
```

### Setup

Open a terminal in the `support_assistant` directory:

```bash
cd support_assistant
pip install -r requirements.txt
```

### Prepare the Knowledge Base

Run the ingestion script:

```bash
python ingest.py
```

This prepares the policy corpus for retrieval.

### Run the API

Start the FastAPI application:

```bash
uvicorn main:app --reload
```

The application exposes:

```text
POST /ask
```

Example request:

```json
{
  "query": "What is the delivery policy?"
}
```

The root endpoint is:

```text
GET /
```

### Design Decisions

The workflow first classifies the customer question and routes policy questions through semantic retrieval. Sentence Transformers provide local embeddings and ChromaDB retrieves the most relevant policy documents. The baseline uses deterministic mock responses so the required implementation can run without a paid external LLM service. Pydantic validates the structured response returned by the API.

### Optional Real LLM

The code contains an optional Groq-based real-LLM path.

It can be enabled with:

```text
MOCK_LLM=0
```

and requires a `GROQ_API_KEY`.

The required baseline does not depend on the real-LLM path.

Detailed architecture and API examples are documented in:

```text
support_assistant/README.md
```

---

## Repository Structure

```text
zepto-data-ai-platform/
├── README.md
├── .gitignore
│
├── data_pipeline/
│   ├── requirements.txt
│   ├── README.md
│   ├── scrape_and_load.py
│   └── run_queries.py
│
├── analytics/
│   ├── requirements.txt
│   ├── README.md
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv
│   ├── cleaned_titanic.csv
│   ├── model_comparison.csv
│   ├── charts/
│   └── models/
│
└── support_assistant/
    ├── requirements.txt
    ├── README.md
    ├── main.py
    ├── ingest.py
    ├── Dockerfile
    └── docs/
```

## Git Workflow

The project uses Git feature-branch workflow.

The repository contains module development and merge history showing work performed on feature branches and merged back into `main`.

The final submission branch is:

```text
main
```

## Academic Integrity

The code, analysis, implementation, and written interpretations in this repository are authored for this project. Standard libraries and framework documentation may be used as technical references.

## Submission

Submit **exactly one public GitHub repository link** for the entire project.

Repository:

https://github.com/suryadevarakonda-ss/zepto-data-ai-platform
