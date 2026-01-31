# B2B Marketplace Intelligence Platform

## Project Overview
This project is a comprehensive data engineering solution designed to scrape, clean, validate, analyze, and visualize data from B2B marketplaces. It features a production-ready pipeline that transforms raw data into actionable business intelligence through advanced statistical analysis and machine learning.

## 🚀 Key Features

### 1. Advanced Data Collection
- **Multi-threaded Scraping:** Optimized concurrent data collection.
- **Robust Error Handling:** Automatic retries, user-agent rotation, and error logging.
- **Mock Data Generator:** Built-in generator for testing and demonstration (5,000+ records).

### 2. Enterprise-Grade Storage
- **Hybrid Storage:** SQLite database for structured querying and CSV for portability.
- **Data Integrity:** Schema validation and duplicate prevention.

### 3. Comprehensive Analysis Engine
- **Statistical Analysis:** Automates descriptive stats, distribution fitting (Weibull, Gamma, etc.), and hypothesis testing (ANOVA).
- **Machine Learning:** Implements K-Means (auto-K selection), DBSCAN, and Hierarchical clustering.
- **Natural Language Processing:** detailed keyword extraction and word cloud generation.

### 4. Premium Visualization Suite
- **Interactive Reports:** Generates a single, self-contained HTML file with 21+ interactive Plotly charts.
- **Dashboard Views:** Executive summary, category deep-dives, and geographic heatmaps.
- **Static Exports:** High-quality Matplotlib charts for static reporting.

## 🛠️ Technical Stack
- **Language:** Python 3.10+
- **Data Processing:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-learn (KMeans, DBSCAN, PCA, t-SNE)
- **Visualization:** Plotly (Interactive), Matplotlib/Seaborn (Static)
- **Infrastructure:** SQLite, SQLAlchemy
- **CLI:** Rich text interface

## 📂 Project Structure
```
Slooze/
├── src/
│   ├── collector.py              # Web scraping & mock data generation
│   ├── database.py               # Database ORM & management
│   ├── models.py                 # Pydantic data models
│   ├── statistics_engine.py      # Advanced statistical analysis (700+ lines)
│   ├── clustering_engine.py      # ML clustering algorithms (600+ lines)
│   ├── visualization_engine.py   # Plotly/Matplotlib engine (1000+ lines)
│   ├── eda_analysis.py          # Main analysis orchestrator
│   └── cli.py                   # Command-line interface
├── data/                        # Storage for DB and CSV files
├── outputs/
│   ├── reports/                 # Generated HTML reports
│   └── charts/                  # Static assets
├── requirements.txt             # Python dependencies
└── README.md                   # Project documentation
```

## 💻 Installation & Usage

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Full Pipeline (Demo Mode)**
   This runs the collector (mock data), analysis, and report generation in one go.
   ```bash
   python src/cli.py pipeline --mock
   ```

3. **View Results**
   Open the generated report: `outputs/reports/complete_analysis_report.html`

## 📊 Analytics Capabilities

| Category | Features |
|----------|----------|
| **Price Analysis** | Distribution tests, Log-transforms, Price spread, Percentile analysis |
| **Clustering** | Customer segmentation via K-Means & DBSCAN, Cluster profiling |
| **Geographic** | Regional price analysis, Market penetration heatmaps |
| **Visualizations** | Funnels, Waterfalls, 3D Scatters, Sunburst charts, Correlation matrices |

## 📝 Sample Insights
The engine automatically detects patterns such as:
- **Price Distribution:** Identifies non-normal distributions and suggests transformations.
- **Correlation:** Detects strong relationships between pricing and verification status.
- **Outliers:** Uses Isolation Forest to flag potential anomalies in pricing data.
- **Segmentation:** Groups products into distinct clusters based on price, reliability, and location.

---
**Author:** Spandana MK
**GitHub:** [Repository Link]
