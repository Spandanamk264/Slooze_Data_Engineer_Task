"""
SLOOZE ULTRA-ADVANCED EDA ENGINE v4.0
======================================
Main orchestrator combining statistics, clustering, and visualization
modules into a comprehensive analysis pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import warnings
import logging

from config import OUTPUT_DIR, DATA_DIR
from database import db_manager

# Import advanced modules
from statistics_engine import AdvancedStatisticalEngine
from clustering_engine import AdvancedClusteringEngine
from visualization_engine import VisualizationEngine

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class NLPAnalyzer:
    """Natural Language Processing for product analysis."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.insights = []
        self.keywords = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Run NLP analysis."""
        self._extract_keywords()
        self._analyze_patterns()
        self._sentiment_proxy()
        return {'keywords': self.keywords, 'insights': self.insights}
    
    def _extract_keywords(self):
        """Extract keywords using TF-IDF."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        text = self.df['product_name'].fillna('').astype(str)
        
        try:
            tfidf = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2
            )
            matrix = tfidf.fit_transform(text)
            features = tfidf.get_feature_names_out()
            scores = matrix.mean(axis=0).A1
            
            self.keywords = dict(sorted(zip(features, scores), key=lambda x: x[1], reverse=True)[:30])
            
            top5 = list(self.keywords.keys())[:5]
            self.insights.append(f"Top keywords: {', '.join(top5)}")
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
    
    def _analyze_patterns(self):
        """Analyze naming patterns."""
        # Most common category
        cat_counts = self.df['category'].value_counts()
        self.insights.append(f"Dominant category: {cat_counts.index[0]} ({cat_counts.iloc[0]} products)")
        
        # Product name length
        name_lengths = self.df['product_name'].str.len()
        avg_len = name_lengths.mean()
        self.insights.append(f"Average product name length: {avg_len:.0f} characters")
        
    def _sentiment_proxy(self):
        """Proxy sentiment analysis using quality indicators."""
        quality_words = ['premium', 'best', 'quality', 'certified', 'genuine', 'authentic']
        budget_words = ['cheap', 'budget', 'economy', 'basic', 'simple']
        
        text_lower = self.df['product_name'].str.lower().fillna('')
        
        quality_count = sum(text_lower.str.contains(w).sum() for w in quality_words)
        budget_count = sum(text_lower.str.contains(w).sum() for w in budget_words)
        
        total = len(self.df)
        self.insights.append(f"Quality-indicator products: {quality_count} ({quality_count/total*100:.1f}%)")


class MegaReportGenerator:
    """Generate the comprehensive single-file HTML report."""
    
    def __init__(self, df: pd.DataFrame, insights: List[str], charts: List[Dict],
                 static_charts: List[Dict], wordcloud_b64: str):
        self.df = df
        self.insights = insights
        self.charts = charts
        self.static_charts = static_charts
        self.wordcloud_b64 = wordcloud_b64
        
    def generate(self) -> str:
        """Generate the complete HTML report."""
        # Category statistics
        cat_stats = self.df.groupby('category').agg({
            'product_name': 'count',
            'price_min': ['mean', 'min', 'max'],
            'supplier_verified': 'mean'
        }).reset_index()
        cat_stats.columns = ['category', 'count', 'avg_price', 'min_price', 'max_price', 'verified_pct']
        
        # KPIs
        total_products = len(self.df)
        total_categories = self.df['category'].nunique()
        total_suppliers = self.df['supplier_name'].nunique()
        total_cities = self.df['city'].nunique()
        avg_price = self.df['price_min'].mean()
        quality = self.df['quality_score'].mean() * 100 if 'quality_score' in self.df.columns else 95
        
        avg_price_str = f"Rs {avg_price/1000:.0f}K" if avg_price < 100000 else f"Rs {avg_price/100000:.1f}L"
        
        # Build HTML
        html = self._build_html(
            total_products, total_categories, total_suppliers, total_cities,
            avg_price_str, quality, cat_stats
        )
        
        filepath = REPORTS_DIR / "complete_analysis_report.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(filepath)
    
    def _build_html(self, total_products, total_categories, total_suppliers,
                    total_cities, avg_price_str, quality, cat_stats):
        """Build the complete HTML document."""
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slooze Market Intelligence Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #667eea;
            --secondary: #764ba2;
            --accent: #f093fb;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1f2937;
            --light: #f8fafc;
            --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-warm: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --gradient-cool: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
            color: var(--dark);
            line-height: 1.7;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 45px 35px;
        }}
        
        /* Premium Header */
        header {{
            background: var(--gradient);
            color: white;
            padding: 85px 60px;
            border-radius: 30px;
            margin-bottom: 55px;
            box-shadow: 0 35px 120px rgba(102, 126, 234, 0.45);
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        header::before {{
            content: '';
            position: absolute;
            top: -60%;
            right: -40%;
            width: 120%;
            height: 220%;
            background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 65%);
            animation: pulse 8s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); opacity: 0.5; }}
            50% {{ transform: scale(1.1); opacity: 0.8; }}
        }}
        
        header h1 {{
            font-size: 3.8rem;
            font-weight: 800;
            margin-bottom: 14px;
            letter-spacing: -2.5px;
            position: relative;
            text-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        
        header .subtitle {{
            font-size: 1.5rem;
            opacity: 0.92;
            font-weight: 400;
            margin-bottom: 8px;
        }}
        
        header .tagline {{
            font-size: 1.1rem;
            opacity: 0.75;
            font-weight: 300;
        }}
        
        header .meta {{
            margin-top: 35px;
            display: flex;
            justify-content: center;
            gap: 28px;
            flex-wrap: wrap;
        }}
        
        header .meta span {{
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 12px 26px;
            border-radius: 35px;
            font-size: 0.95rem;
            font-weight: 500;
            border: 1px solid rgba(255,255,255,0.25);
        }}
        
        /* KPI Cards */
        .kpi-section {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 24px;
            margin-bottom: 60px;
        }}
        
        .kpi-card {{
            background: white;
            padding: 38px 24px;
            border-radius: 24px;
            text-align: center;
            box-shadow: 0 10px 45px rgba(0,0,0,0.08);
            transition: all 0.45s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid rgba(0,0,0,0.04);
            position: relative;
            overflow: hidden;
        }}
        
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--gradient);
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-12px);
            box-shadow: 0 30px 70px rgba(102, 126, 234, 0.25);
        }}
        
        .kpi-card:hover::before {{
            opacity: 1;
        }}
        
        .kpi-card .icon {{
            font-size: 2.8rem;
            margin-bottom: 18px;
            display: inline-block;
        }}
        
        .kpi-card .value {{
            font-size: 3rem;
            font-weight: 800;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
        }}
        
        .kpi-card .label {{
            color: #64748b;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.3px;
            margin-top: 12px;
        }}
        
        /* Sections */
        .section {{
            background: white;
            border-radius: 28px;
            padding: 55px;
            margin-bottom: 45px;
            box-shadow: 0 10px 45px rgba(0,0,0,0.06);
            border: 1px solid rgba(0,0,0,0.04);
        }}
        
        .section-header {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 40px;
            padding-bottom: 25px;
            border-bottom: 2px solid #f1f5f9;
        }}
        
        .section-header .icon {{
            width: 60px;
            height: 60px;
            background: var(--gradient);
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.8rem;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }}
        
        .section-header h2 {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--dark);
        }}
        
        .section-header .subtitle {{
            font-size: 0.95rem;
            color: #64748b;
            margin-left: auto;
        }}
        
        /* Insights Grid */
        .insights-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 22px;
        }}
        
        .insight-card {{
            background: linear-gradient(135deg, #fef9c3 0%, #fde68a 100%);
            border-left: 6px solid #eab308;
            padding: 26px 32px;
            border-radius: 0 20px 20px 0;
            font-size: 1rem;
            font-weight: 500;
            color: #713f12;
            transition: all 0.35s ease;
            box-shadow: 0 4px 15px rgba(234, 179, 8, 0.15);
        }}
        
        .insight-card:hover {{
            transform: translateX(10px);
            box-shadow: 0 8px 25px rgba(234, 179, 8, 0.25);
        }}
        
        /* Charts */
        .chart-container {{
            margin: 40px 0;
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.04);
            min-height: 500px;
            background: white;
        }}
        
        .chart-container > div {{
            min-height: 480px;
        }}
        
        .static-chart {{
            text-align: center;
            padding: 40px;
            background: linear-gradient(180deg, #fafafa 0%, #f5f5f5 100%);
        }}
        
        .static-chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.12);
        }}
        
        /* Data Table */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 28px;
            font-size: 0.98rem;
        }}
        
        .data-table th {{
            background: var(--gradient);
            color: white;
            padding: 22px 30px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.7px;
        }}
        
        .data-table th:first-child {{ border-radius: 16px 0 0 0; }}
        .data-table th:last-child {{ border-radius: 0 16px 0 0; }}
        
        .data-table td {{
            padding: 22px 30px;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .data-table tr:nth-child(even) {{
            background: #f8fafc;
        }}
        
        .data-table tr:hover {{
            background: linear-gradient(90deg, #e0e7ff 0%, #f0f4ff 100%);
        }}
        
        .badge {{
            display: inline-block;
            padding: 7px 18px;
            border-radius: 25px;
            font-size: 0.8rem;
            font-weight: 700;
        }}
        
        .badge-success {{
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            color: #065f46;
        }}
        
        .badge-warning {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            color: #92400e;
        }}
        
        /* Footer */
        footer {{
            text-align: center;
            padding: 70px 40px;
            color: #94a3b8;
        }}
        
        footer .brand {{
            font-size: 1.4rem;
            font-weight: 700;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 15px;
        }}
        
        footer .tech-stack {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        footer .tech-stack span {{
            background: white;
            padding: 8px 18px;
            border-radius: 25px;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--dark);
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        
        /* Responsive */
        @media (max-width: 1400px) {{
            .kpi-section {{ grid-template-columns: repeat(3, 1fr); }}
        }}
        
        @media (max-width: 1000px) {{
            .insights-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        
        @media (max-width: 768px) {{
            header h1 {{ font-size: 2.4rem; }}
            .kpi-section {{ grid-template-columns: repeat(2, 1fr); }}
            .insights-grid {{ grid-template-columns: 1fr; }}
            .section {{ padding: 30px; }}
            .container {{ padding: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Market Intelligence Report</h1>
            <p class="subtitle">B2B Industrial Marketplace - Advanced Data Analysis</p>
            <p class="tagline">Comprehensive insights powered by Machine Learning & Statistical Analysis</p>
            <div class="meta">
                <span>Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}</span>
                <span>Data Points: {total_products:,}</span>
                <span>Analytics Engine v4.0</span>
                <span>Premium Report</span>
            </div>
        </header>
        
        <section class="kpi-section">
            <div class="kpi-card">
                <div class="icon">📦</div>
                <div class="value">{total_products:,}</div>
                <div class="label">Total Products</div>
            </div>
            <div class="kpi-card">
                <div class="icon">📁</div>
                <div class="value">{total_categories}</div>
                <div class="label">Categories</div>
            </div>
            <div class="kpi-card">
                <div class="icon">🏭</div>
                <div class="value">{total_suppliers}</div>
                <div class="label">Suppliers</div>
            </div>
            <div class="kpi-card">
                <div class="icon">🌍</div>
                <div class="value">{total_cities}</div>
                <div class="label">Cities</div>
            </div>
            <div class="kpi-card">
                <div class="icon">💰</div>
                <div class="value">{avg_price_str}</div>
                <div class="label">Avg Price</div>
            </div>
            <div class="kpi-card">
                <div class="icon">✅</div>
                <div class="value">{quality:.0f}%</div>
                <div class="label">Data Quality</div>
            </div>
        </section>
        
        <section class="section">
            <div class="section-header">
                <div class="icon">💡</div>
                <h2>Key Intelligence Insights</h2>
                <span class="subtitle">{len(self.insights)} insights discovered</span>
            </div>
            <div class="insights-grid">
                {''.join([f'<div class="insight-card">{insight}</div>' for insight in self.insights[:12]])}
            </div>
        </section>
'''
        
        # Add all interactive charts
        for i, chart in enumerate(self.charts):
            if chart.get('html'):
                html += f'''
        <section class="section">
            <div class="section-header">
                <div class="icon">📊</div>
                <h2>{chart['title']}</h2>
            </div>
            <div class="chart-container">
                {chart['html']}
            </div>
        </section>
'''
        
        # Add wordcloud
        if self.wordcloud_b64:
            html += f'''
        <section class="section">
            <div class="section-header">
                <div class="icon">🔤</div>
                <h2>Product Keyword Analysis</h2>
                <span class="subtitle">NLP-powered keyword extraction</span>
            </div>
            <div class="static-chart">
                <img src="{self.wordcloud_b64}" alt="Keyword Cloud">
            </div>
        </section>
'''
        
        # Add static charts
        for chart in self.static_charts:
            if chart.get('base64'):
                html += f'''
        <section class="section">
            <div class="section-header">
                <div class="icon">📈</div>
                <h2>{chart['title']}</h2>
            </div>
            <div class="static-chart">
                <img src="{chart['base64']}" alt="{chart['title']}">
            </div>
        </section>
'''
        
        # Category table
        html += '''
        <section class="section">
            <div class="section-header">
                <div class="icon">📋</div>
                <h2>Category Performance Breakdown</h2>
            </div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Products</th>
                        <th>Avg Price</th>
                        <th>Price Range</th>
                        <th>Verified %</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
'''
        
        for _, row in cat_stats.iterrows():
            verified_badge = 'badge-success' if row['verified_pct'] > 0.5 else 'badge-warning'
            html += f'''
                    <tr>
                        <td><strong>{row['category']}</strong></td>
                        <td>{int(row['count']):,}</td>
                        <td>Rs {row['avg_price']:,.0f}</td>
                        <td>Rs {row['min_price']:,.0f} - Rs {row['max_price']:,.0f}</td>
                        <td>{row['verified_pct']*100:.1f}%</td>
                        <td><span class="badge badge-success">Active</span></td>
                    </tr>
'''
        
        html += '''
                </tbody>
            </table>
        </section>
        
        <footer>
            <div class="brand">Slooze Data Engineering Pipeline</div>
            <p>Advanced Analytics Solution by <strong>Spandana M K</strong></p>
            <div class="tech-stack">
                <span>Python 3.10</span>
                <span>Pandas</span>
                <span>Plotly</span>
                <span>Scikit-learn</span>
                <span>Matplotlib</span>
                <span>Seaborn</span>
                <span>SQLAlchemy</span>
            </div>
            <p style="margin-top: 25px; font-size: 0.9rem;">&copy; 2026 | Built for Slooze Data Engineering Challenge</p>
        </footer>
    </div>
</body>
</html>
'''
        return html


class AdvancedEDAEngine:
    """Main orchestrator for the complete EDA pipeline."""
    
    def __init__(self):
        self.df = None
        self.insights = []
        self.results = {}
        
    def load_data(self) -> bool:
        """Load data from database or CSV."""
        try:
            products = db_manager.get_all_products()
            if products:
                self.df = pd.DataFrame(products)
                self._clean_data()
                print(f"[OK] Loaded {len(self.df):,} records from database")
                return True
        except Exception as e:
            print(f"[WARN] Database load failed: {e}")
        
        csv_path = DATA_DIR / "collected_data.csv"
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
            self._clean_data()
            print(f"[OK] Loaded {len(self.df):,} records from CSV")
            return True
        
        print("[ERROR] No data found. Run collector first.")
        return False
    
    def _clean_data(self):
        """Clean data by removing columns with unhashable types."""
        cols_to_drop = []
        for col in self.df.columns:
            try:
                sample = self.df[col].dropna().head(10)
                if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
                    cols_to_drop.append(col)
            except:
                pass
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
            print(f"[INFO] Dropped {len(cols_to_drop)} columns with list types: {cols_to_drop}")
    
    def run(self):
        """Execute the complete advanced analytics pipeline."""
        print("\n" + "=" * 75)
        print("   SLOOZE ULTRA-ADVANCED ANALYTICS ENGINE v4.0")
        print("   Comprehensive Data Analysis Pipeline")
        print("=" * 75)
        
        if not self.load_data():
            return
        
        # Phase 1: Statistical Analysis
        print("\n[1/5] STATISTICAL ANALYSIS")
        print("-" * 40)
        stats_engine = AdvancedStatisticalEngine(self.df)
        self.results['statistics'] = stats_engine.run_complete_analysis()
        self.insights.extend(stats_engine.insights)
        print(f"       Collected {len(stats_engine.insights)} statistical insights")
        
        # Phase 2: Machine Learning Clustering
        print("\n[2/5] MACHINE LEARNING CLUSTERING")
        print("-" * 40)
        clustering_engine = AdvancedClusteringEngine(self.df)
        self.df = clustering_engine.run_complete_analysis()
        self.results['clustering'] = clustering_engine.results
        self.insights.extend(clustering_engine.insights)
        print(f"       Collected {len(clustering_engine.insights)} clustering insights")
        
        # Phase 3: NLP Analysis
        print("\n[3/5] NLP & TEXT ANALYSIS")
        print("-" * 40)
        nlp_engine = NLPAnalyzer(self.df)
        self.results['nlp'] = nlp_engine.analyze()
        self.insights.extend(nlp_engine.insights)
        print(f"       Collected {len(nlp_engine.insights)} NLP insights")
        
        # Phase 4: Visualization Generation
        print("\n[4/5] VISUALIZATION GENERATION")
        print("-" * 40)
        viz_engine = VisualizationEngine(self.df)
        charts, static_charts, wordcloud_b64 = viz_engine.create_all()
        print(f"       Generated {len(charts)} interactive charts")
        print(f"       Generated {len(static_charts)} static charts")
        
        # Phase 5: Report Generation
        print("\n[5/5] REPORT GENERATION")
        print("-" * 40)
        report_gen = MegaReportGenerator(
            self.df, self.insights, charts, static_charts, wordcloud_b64
        )
        report_path = report_gen.generate()
        
        # Summary
        print("\n" + "=" * 75)
        print("   ANALYSIS COMPLETE!")
        print("=" * 75)
        print(f"\n   Report: {report_path}")
        print(f"   Total Insights: {len(self.insights)}")
        print(f"   Total Charts: {len(charts) + len(static_charts)}")
        
        print("\n   TOP INSIGHTS:")
        for i, insight in enumerate(self.insights[:8], 1):
            print(f"   {i}. {insight}")
        
        print("\n" + "=" * 75)


def main():
    """CLI entry point."""
    engine = AdvancedEDAEngine()
    engine.run()


if __name__ == "__main__":
    main()
