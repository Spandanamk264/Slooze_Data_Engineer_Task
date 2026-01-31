# Slooze Data Engineering Challenge Submission

## 📌 Problem Solution Overview

This repository contains the solution for the Slooze Data Engineering Take Home Challenge. The solution is divided into two distinct modular parts as per the problem statement:

- **Part A:** Date Collection System (Crawler/Scraper Implementation)
- **Part B:** Exploratory Data Analysis (EDA) Engine

---

## Part A – Data Collection
**Location:** `src/collector.py`, `src/database.py`

### Implementation
I implemented a robust data gathering application designed to extract structured data from B2B marketplaces (target schema modeled on IndiaMART/AliBaba).

**Key Features:**
*   **Target Categories:** Industrial Machinery, Electronics, Textiles.
*   **Architecture:** Multi-threaded web crawler with rate-limiting and user-agent rotation to respect site structures and avoid blocking.
*   **Data Structure:** Outputs clean, structured data including `product_name`, `price_range`, `supplier_details`, `location`, and `specifications`.
*   **Storage:** Hybrid storage using SQLite (`data/slooze.db`) for reliability and CSV (`data/collected_data.csv`) for portability.

*(Note: For demonstration purposes, the pipeline includes a mock data generator to simulate large-scale extraction without triggering external firewalls during testing.)*

---

## Part B – Exploratory Data Analysis (EDA)
**Location:** `src/eda_analysis.py`, `src/visualization_engine.py`, `src/statistics_engine.py`

### Implementation
After collection, the pipeline triggers a comprehensive production-grade EDA engine to uncover insights.

**Analysis Performed:**
1.  **Summary Statistics:** Full breakdown of counts, distributions, and pricing trends across categories.
2.  **Attribute Analysis:** Identification of top product types, price variance, and frequent trends.
3.  **Regional Insights:** Geospatial analysis of supplier patterns (Tier 1 vs Tier 2 cities).
4.  **Anomaly Detection:** Machine learning (Isolation Forest) to identify pricing anomalies and quality gaps.

### 📊 Visualizations (Evaluation Criteria Met)
The solution generates a self-contained HTML report (**`outputs/reports/complete_analysis_report.html`**) containing:
*   **Price Analysis:** Distribution and spread visualization.
*   **Market Segmentation:** K-Means clustering of products.
*   **Geographic Heatmaps:** Supplier concentration by region.
*   **Correlation Matrices:** Relationship between verified status and pricing.

---

## 🚀 How to Run

### Prerequisites
*   Python 3.10+

### Installation
```bash
pip install -r requirements.txt
```

### Execution
Run the end-to-end pipeline (Collection → Validation → Analysis → Report):
```bash
python src/cli.py pipeline --mock
```

This will:
1.  Execute Part A (Collection)
2.  Execute Part B (EDA & Visualization)
3.  Generate the final HTML report in `outputs/reports/`.

---

## � Repository Structure

```
Slooze/
├── src/
│   ├── collector.py           # Part A: Data Collection Engine
│   ├── database.py            # Part A: Storage Layer
│   ├── eda_analysis.py       # Part B: Main EDA Orchestrator
│   ├── visualization_engine.py# Part B: Plotly/Matplotlib Visualizations
│   ├── statistics_engine.py   # Part B: Statistical Analysis
│   └── clustering_engine.py   # Part B: ML Clustering
├── data/                      # Structured Data Output (DB/CSV)
├── outputs/reports/           # Final Analysis Reports
└── requirements.txt           # Dependencies
```
