<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-Private-red.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen.svg" alt="Status">
</p>

<h1 align="center">Slooze Data Engineering Challenge</h1>

<p align="center">
  <strong>Production-Grade B2B Marketplace Intelligence Pipeline</strong>
</p>

<p align="center">
  <em>A comprehensive data collection and analysis solution demonstrating advanced data engineering practices</em>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Sample Outputs](#sample-outputs)
- [Design Decisions](#design-decisions)
- [Author](#author)

---

## Overview

This repository contains my solution for the **Slooze Data Engineering Take-Home Challenge**, encompassing:

- **Part A**: Advanced web scraping and data collection from B2B marketplaces
- **Part B**: Comprehensive Exploratory Data Analysis with interactive visualizations

The solution is designed with production-readiness in mind, featuring robust error handling, data validation, and scalable architecture patterns.

---

## Features

### Data Collection Engine

| Feature | Description |
|---------|-------------|
| **Asynchronous Scraping** | High-performance concurrent requests using `aiohttp` |
| **Intelligent Rate Limiting** | Token bucket algorithm to respect target site policies |
| **User-Agent Rotation** | Automatic rotation to avoid detection |
| **Retry Logic** | Exponential backoff with configurable attempts |
| **Schema Validation** | Pydantic models ensure data integrity at ingestion |
| **Quality Scoring** | Automated data completeness assessment |
| **Resilient Fallback** | Mock data generation when scraping is blocked |

### Analytics & Visualization

| Feature | Description |
|---------|-------------|
| **Statistical Analysis** | Distribution analysis, outlier detection, correlation matrices |
| **Market Segmentation** | K-Means clustering for price-based product segments |
| **NLP Processing** | TF-IDF keyword extraction and word cloud generation |
| **Interactive Dashboards** | Plotly-powered charts with hover details and filtering |
| **Executive Reports** | Auto-generated HTML reports with key insights |
| **Regional Analysis** | Geographic distribution of suppliers and products |

### Engineering Excellence

| Feature | Description |
|---------|-------------|
| **Type Safety** | Full type hints with Pydantic validation |
| **Database Layer** | SQLAlchemy ORM with batch operations |
| **CLI Interface** | Rich terminal UI with progress indicators |
| **Structured Logging** | Comprehensive logging for debugging |
| **Modular Design** | Clean separation of concerns |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI Application                              │
│                      (Typer + Rich Console)                          │
└───────────────────┬─────────────────────────┬───────────────────────┘
                    │                         │
                    ▼                         ▼
      ┌─────────────────────────┐   ┌─────────────────────────────┐
      │    Data Collector       │   │      EDA Engine             │
      │                         │   │                             │
      │  ┌─────────────────┐    │   │  ┌───────────────────────┐  │
      │  │ Async HTTP      │    │   │  │ Statistical Analyzer  │  │
      │  │ Rate Limiter    │    │   │  │ Visualization Engine  │  │
      │  │ Parser Factory  │    │   │  │ NLP Processor         │  │
      │  │ Mock Generator  │    │   │  │ Clustering Engine     │  │
      │  └─────────────────┘    │   │  │ Report Generator      │  │
      │                         │   │  └───────────────────────┘  │
      │  ┌─────────────────┐    │   │                             │
      │  │ Pydantic Models │    │   │                             │
      │  │ (Validation)    │    │   │                             │
      │  └─────────────────┘    │   │                             │
      └───────────┬─────────────┘   └──────────────┬──────────────┘
                  │                                │
                  ▼                                ▼
      ┌───────────────────────────────────────────────────────────┐
      │                    Database Layer                          │
      │                  (SQLAlchemy ORM)                          │
      │  ┌─────────────────────────────────────────────────────┐   │
      │  │              SQLite Database                        │   │
      │  │   • Products Table (indexed)                        │   │
      │  │   • Scraping Runs Table (audit log)                 │   │
      │  └─────────────────────────────────────────────────────┘   │
      └───────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Category | Technologies |
|----------|--------------|
| **Core Language** | Python 3.8+ |
| **Async HTTP** | aiohttp, asyncio |
| **Web Scraping** | BeautifulSoup4, lxml |
| **Data Validation** | Pydantic |
| **Database** | SQLite, SQLAlchemy |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Machine Learning** | Scikit-learn |
| **NLP** | NLTK, WordCloud |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **CLI Framework** | Typer, Rich |
| **Templating** | Jinja2 |
| **Resilience** | Tenacity |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/Spandanamk264/Slooze_Data_Engineer_Task.git
cd Slooze_Data_Engineer_Task

# Create virtual environment
python -m venv env

# Activate virtual environment
# Windows:
.\env\Scripts\activate
# Linux/macOS:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Quick Start (Recommended)

Run the complete pipeline with a single command:

```bash
python src/cli.py pipeline --mock
```

This will:
1. Generate realistic mock data (300 products across 5 categories)
2. Validate and store data in SQLite database
3. Run comprehensive EDA
4. Generate interactive visualizations
5. Create HTML analysis report

### Individual Commands

```bash
# Data Collection only
python src/cli.py collect --mock

# Analysis only (requires data)
python src/cli.py analyze

# Check pipeline status
python src/cli.py status

# Clear all data
python src/cli.py clean
```

### Command Options

| Command | Flag | Description |
|---------|------|-------------|
| `collect` | `--mock` | Use mock data generation instead of live scraping |
| `pipeline` | `--mock` | Run full pipeline with mock data |
| `status` | - | Display current data statistics |
| `clean` | - | Remove all collected data and outputs |

---

## Project Structure

```
Slooze_Data_Engineer_Task/
│
├── src/                          # Source code
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Configuration management
│   ├── models.py                 # Pydantic data models
│   ├── database.py               # SQLAlchemy ORM layer
│   ├── collector.py              # Data collection engine
│   └── eda_analysis.py           # EDA and visualization engine
│
├── data/                         # Data storage
│   ├── slooze_data.db            # SQLite database
│   └── collected_data.csv        # CSV export
│
├── outputs/                      # Generated outputs
│   ├── charts/                   # Visualization files
│   │   ├── executive_summary.png
│   │   ├── price_distribution_interactive.html
│   │   ├── regional_analysis_interactive.html
│   │   ├── supplier_analysis_interactive.html
│   │   ├── category_sunburst.html
│   │   ├── price_clusters.html
│   │   └── wordcloud.png
│   │
│   └── reports/                  # Generated reports
│       └── analysis_report.html
│
├── CHALLENGE_DETAILS.md          # Original challenge description
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## Sample Outputs

### Executive Summary Dashboard
The pipeline generates a 4-panel executive summary including:
- Price distribution with KDE
- Top manufacturing hubs
- Category price comparison
- Supplier verification status

### Interactive Visualizations
All interactive charts are saved as standalone HTML files that can be opened in any browser:
- **Price Distribution**: Histogram with category breakdown
- **Regional Analysis**: City-wise product and price comparison
- **Supplier Analysis**: Top suppliers with verification status
- **Category Sunburst**: Hierarchical category-city distribution
- **Price Clusters**: K-Means segmentation scatter plot

### HTML Report
A comprehensive analysis report with:
- Key statistics (total products, categories, suppliers, cities)
- Data quality score
- Auto-generated business insights
- Links to all interactive visualizations
- Category-wise statistics table

---

## Design Decisions

### Why Async Scraping?
Asynchronous HTTP requests allow concurrent data fetching, significantly improving performance while respecting rate limits through the token bucket algorithm.

### Why Pydantic for Validation?
Pydantic provides runtime type checking and automatic data coercion, ensuring data quality at the point of ingestion rather than discovering issues during analysis.

### Why SQLite?
For a take-home challenge, SQLite provides the benefits of a proper database (ACID compliance, indexing, SQL queries) without requiring external setup. The SQLAlchemy abstraction allows easy migration to PostgreSQL or MySQL for production.

### Why Mock Data Fallback?
B2B marketplaces often block scrapers. The mock data generator ensures the pipeline is always demonstrable, producing statistically realistic data distributions that enable meaningful EDA.

### Why Interactive Visualizations?
Plotly's interactive charts allow stakeholders to explore data through hover details, zoom, and filtering—providing more value than static images alone.

---

## Author

**Spandana M K**  
Data Engineering Candidate

---

<p align="center">
  <em>Built with precision and passion for data engineering excellence</em>
</p>
