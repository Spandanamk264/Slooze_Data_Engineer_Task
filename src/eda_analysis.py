"""
Slooze Data Engineering Challenge - Advanced EDA & Analytics
=============================================================

Comprehensive exploratory data analysis with:
- Statistical analysis
- Interactive visualizations
- NLP text analysis
- Clustering and segmentation
- Automated insight generation
- HTML report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
import warnings
from pathlib import Path
from jinja2 import Template

from config import OUTPUT_DIR, analysis_config, DATA_DIR
from database import db_manager

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set premium style
plt.style.use(analysis_config.CHART_STYLE)
sns.set_palette(analysis_config.COLOR_PALETTE)

# Output directories
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"


class StatisticalAnalyzer:
    """Advanced statistical analysis."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.insights: List[str] = []
        
    def compute_summary_stats(self) -> Dict[str, Any]:
        """Compute comprehensive summary statistics."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        stats_dict = {}
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                stats_dict[col] = {
                    'count': len(data),
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'median': data.median(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'iqr': data.quantile(0.75) - data.quantile(0.25),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data)
                }
        
        return stats_dict
    
    def detect_outliers(self, column: str, method: str = 'iqr') -> pd.Series:
        """Detect outliers using IQR or Z-score method."""
        data = self.df[column].dropna()
        
        if method == 'iqr':
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (data < lower) | (data > upper)
        else:
            z_scores = np.abs(stats.zscore(data))
            return z_scores > 3
    
    def correlation_analysis(self) -> pd.DataFrame:
        """Compute correlation matrix for numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.corr()
    
    def price_distribution_analysis(self) -> Dict[str, Any]:
        """Analyze price distribution characteristics."""
        if 'price_min' not in self.df.columns:
            return {}
        
        prices = self.df['price_min'].dropna()
        
        # Test for normality
        if len(prices) > 8:
            _, normality_p = stats.normaltest(prices)
        else:
            normality_p = None
        
        # Distribution fit
        result = {
            'is_normal': normality_p > 0.05 if normality_p else None,
            'normality_p_value': normality_p,
            'distribution_type': 'log-normal' if prices.skew() > 1 else 'approximately normal',
            'price_segments': self._segment_prices(prices)
        }
        
        # Generate insight
        if prices.skew() > 1:
            self.insights.append(
                f"[Price] Distribution is right-skewed (skewness: {prices.skew():.2f}), "
                "indicating most products are in lower price ranges with some premium items."
            )
        
        return result
    
    def _segment_prices(self, prices: pd.Series) -> Dict[str, int]:
        """Segment prices into buckets."""
        bins = [0, 10000, 50000, 200000, 1000000, float('inf')]
        labels = ['Budget (<10K)', 'Economy (10K-50K)', 'Mid-range (50K-2L)', 
                  'Premium (2L-10L)', 'Enterprise (>10L)']
        
        segments = pd.cut(prices, bins=bins, labels=labels)
        return segments.value_counts().to_dict()


class VisualizationEngine:
    """Advanced visualization generation."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.color_palette = px.colors.qualitative.Set2
        
    def create_price_distribution(self) -> Tuple[go.Figure, str]:
        """Create interactive price distribution chart."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Price Distribution', 'Price by Category'),
            specs=[[{"type": "histogram"}, {"type": "box"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=self.df['price_min'],
                nbinsx=30,
                marker_color='#667eea',
                opacity=0.7,
                name='Price Distribution'
            ),
            row=1, col=1
        )
        
        # Box plot by category
        categories = self.df['category'].unique()
        for i, cat in enumerate(categories):
            cat_data = self.df[self.df['category'] == cat]['price_min']
            fig.add_trace(
                go.Box(
                    y=cat_data,
                    name=cat[:15],
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text='<b>Price Analysis Dashboard</b>',
            title_font_size=20,
            showlegend=False,
            template='plotly_white',
            height=500
        )
        
        filepath = CHARTS_DIR / "price_distribution_interactive.html"
        fig.write_html(str(filepath))
        
        return fig, str(filepath)
    
    def create_regional_heatmap(self) -> Tuple[go.Figure, str]:
        """Create regional insights visualization."""
        # Aggregate by city
        city_stats = self.df.groupby('city').agg({
            'product_name': 'count',
            'price_min': 'mean'
        }).rename(columns={'product_name': 'count', 'price_min': 'avg_price'})
        city_stats = city_stats.reset_index().sort_values('count', ascending=False).head(15)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Products by City', 'Average Price by City'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Count bar
        fig.add_trace(
            go.Bar(
                x=city_stats['city'],
                y=city_stats['count'],
                marker_color='#764ba2',
                name='Product Count'
            ),
            row=1, col=1
        )
        
        # Price bar
        fig.add_trace(
            go.Bar(
                x=city_stats['city'],
                y=city_stats['avg_price'],
                marker_color='#f093fb',
                name='Avg Price'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text='<b>Regional Market Analysis</b>',
            title_font_size=20,
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45)
        
        filepath = CHARTS_DIR / "regional_analysis_interactive.html"
        fig.write_html(str(filepath))
        
        return fig, str(filepath)
    
    def create_supplier_analysis(self) -> Tuple[go.Figure, str]:
        """Create supplier insights visualization."""
        supplier_stats = self.df.groupby('supplier_name').agg({
            'product_name': 'count',
            'price_min': 'mean',
            'supplier_verified': 'first'
        }).rename(columns={'product_name': 'listings'})
        
        supplier_stats = supplier_stats.reset_index().nlargest(10, 'listings')
        
        fig = go.Figure()
        
        colors = ['#2ecc71' if v else '#e74c3c' for v in supplier_stats['supplier_verified']]
        
        fig.add_trace(go.Bar(
            x=supplier_stats['supplier_name'],
            y=supplier_stats['listings'],
            marker_color=colors,
            text=supplier_stats['listings'],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Listings: %{y}<br>Avg Price: ₹%{customdata:,.0f}',
            customdata=supplier_stats['price_min']
        ))
        
        fig.update_layout(
            title_text='<b>Top Suppliers Analysis</b><br><sub>Green = Verified | Red = Unverified</sub>',
            title_font_size=18,
            xaxis_title='Supplier',
            yaxis_title='Number of Listings',
            template='plotly_white',
            height=500
        )
        
        fig.update_xaxes(tickangle=45)
        
        filepath = CHARTS_DIR / "supplier_analysis_interactive.html"
        fig.write_html(str(filepath))
        
        return fig, str(filepath)
    
    def create_category_sunburst(self) -> Tuple[go.Figure, str]:
        """Create category hierarchy sunburst chart."""
        # Create hierarchy data
        hierarchy_data = self.df.groupby(['category', 'city']).size().reset_index(name='count')
        
        fig = px.sunburst(
            hierarchy_data,
            path=['category', 'city'],
            values='count',
            color='count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title_text='<b>Category-City Distribution</b>',
            title_font_size=20,
            height=600
        )
        
        filepath = CHARTS_DIR / "category_sunburst.html"
        fig.write_html(str(filepath))
        
        return fig, str(filepath)
    
    def create_static_charts(self):
        """Create static matplotlib charts for PDF/report embedding."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Slooze Data Analysis - Executive Summary', fontsize=20, fontweight='bold')
        
        # 1. Price distribution with KDE
        ax1 = axes[0, 0]
        sns.histplot(data=self.df, x='price_min', kde=True, ax=ax1, color='#667eea')
        ax1.set_title('Price Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Price (INR)')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
        
        # 2. Top cities bar chart
        ax2 = axes[0, 1]
        top_cities = self.df['city'].value_counts().head(10)
        colors = sns.color_palette('viridis', len(top_cities))
        bars = ax2.barh(top_cities.index[::-1], top_cities.values[::-1], color=colors)
        ax2.set_title('Top 10 Manufacturing Hubs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Products')
        for bar, val in zip(bars, top_cities.values[::-1]):
            ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), 
                    va='center', fontsize=10)
        
        # 3. Category price comparison
        ax3 = axes[1, 0]
        category_order = self.df.groupby('category')['price_min'].median().sort_values().index
        sns.boxplot(data=self.df, x='category', y='price_min', ax=ax3, 
                   order=category_order, palette='viridis')
        ax3.set_title('Price Range by Category', fontsize=14, fontweight='bold')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
        ax3.set_xlabel('')
        ax3.set_ylabel('Price (INR)')
        
        # 4. Verified vs Unverified suppliers
        ax4 = axes[1, 1]
        verified_counts = self.df['supplier_verified'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        labels = ['Verified', 'Unverified']
        explode = (0.05, 0)
        wedges, texts, autotexts = ax4.pie(
            verified_counts.values, 
            labels=labels,
            colors=colors,
            explode=explode,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        ax4.set_title('Supplier Verification Status', fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        filepath = CHARTS_DIR / "executive_summary.png"
        plt.savefig(filepath, dpi=analysis_config.FIGURE_DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)


class NLPAnalyzer:
    """Natural Language Processing for product analysis."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def generate_word_cloud(self) -> str:
        """Generate word cloud from product names."""
        text = ' '.join(self.df['product_name'].dropna().astype(str))
        
        # Custom stopwords
        stopwords = {'model', 'type', 'new', 'quality', 'best', 'for', 'the', 'and', 
                    'with', 'pro', 'standard', 'grade', 'industrial', 'machine'}
        
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis',
            max_words=100,
            stopwords=stopwords,
            min_font_size=10,
            max_font_size=100
        ).generate(text.lower())
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Product Name Word Cloud', fontsize=20, fontweight='bold', pad=20)
        
        filepath = CHARTS_DIR / "wordcloud.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(filepath)
    
    def extract_keywords(self) -> Dict[str, int]:
        """Extract top keywords using TF-IDF."""
        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(self.df['product_name'].fillna(''))
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            scores = tfidf_matrix.mean(axis=0).A1
            keywords = dict(zip(feature_names, scores))
            
            return dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:20])
        except:
            return {}


class ClusteringEngine:
    """Product clustering and segmentation."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def perform_price_clustering(self, n_clusters: int = 5) -> pd.DataFrame:
        """Cluster products by price characteristics."""
        features = self.df[['price_min', 'price_max']].dropna()
        
        if len(features) < n_clusters:
            return self.df
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Map cluster labels to meaningful names
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        sorted_indices = np.argsort(cluster_centers[:, 0])
        
        label_map = {
            sorted_indices[0]: 'Budget',
            sorted_indices[1]: 'Economy',
            sorted_indices[2]: 'Mid-Range',
            sorted_indices[3]: 'Premium',
            sorted_indices[4]: 'Enterprise' if n_clusters > 4 else 'Premium'
        }
        
        self.df.loc[features.index, 'price_segment'] = [label_map.get(c, f'Cluster {c}') for c in clusters]
        
        return self.df
    
    def visualize_clusters(self) -> str:
        """Visualize price clusters."""
        if 'price_segment' not in self.df.columns:
            self.perform_price_clustering()
        
        fig = px.scatter(
            self.df.dropna(subset=['price_min', 'price_max', 'price_segment']),
            x='price_min',
            y='price_max',
            color='price_segment',
            hover_data=['product_name', 'supplier_name', 'city'],
            title='<b>Product Price Segmentation</b>',
            labels={'price_min': 'Minimum Price (INR)', 'price_max': 'Maximum Price (INR)'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(template='plotly_white', height=600)
        
        filepath = CHARTS_DIR / "price_clusters.html"
        fig.write_html(str(filepath))
        
        return str(filepath)


class ReportGenerator:
    """Generate comprehensive HTML report."""
    
    REPORT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slooze Data Analysis Report</title>
    <style>
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --success: #2ecc71;
            --danger: #e74c3c;
            --dark: #2c3e50;
            --light: #ecf0f1;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        header .subtitle {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card .value {
            font-size: 2rem;
            font-weight: bold;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-card .label {
            color: #666;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        
        .section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        
        .section h2 {
            color: var(--primary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--light);
        }
        
        .insights {
            background: linear-gradient(135deg, #fff9e6 0%, #fff3cd 100%);
            border-left: 4px solid #f39c12;
            padding: 20px;
            border-radius: 0 10px 10px 0;
            margin: 15px 0;
        }
        
        .insights h3 {
            color: #f39c12;
            margin-bottom: 15px;
        }
        
        .insights ul {
            list-style: none;
        }
        
        .insights li {
            padding: 8px 0;
            border-bottom: 1px dashed #ddd;
        }
        
        .insights li:last-child {
            border-bottom: none;
        }
        
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .chart-link {
            display: inline-block;
            margin: 10px;
            padding: 10px 20px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            text-decoration: none;
            border-radius: 25px;
            transition: transform 0.3s ease;
        }
        
        .chart-link:hover {
            transform: scale(1.05);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background: var(--primary);
            color: white;
        }
        
        tr:hover {
            background: #f5f5f5;
        }
        
        footer {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        @media (max-width: 768px) {
            header h1 {
                font-size: 1.8rem;
            }
            
            .stats-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Slooze Data Analysis Report</h1>
            <p class="subtitle">B2B Marketplace Intelligence | Generated: {{ generated_at }}</p>
        </header>
        
        <section class="stats-grid">
            <div class="stat-card">
                <div class="value">{{ total_records }}</div>
                <div class="label">Total Products</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ categories_count }}</div>
                <div class="label">Categories</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ suppliers_count }}</div>
                <div class="label">Unique Suppliers</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ cities_count }}</div>
                <div class="label">Cities Covered</div>
            </div>
            <div class="stat-card">
                <div class="value">₹{{ avg_price }}</div>
                <div class="label">Average Price</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ quality_score }}%</div>
                <div class="label">Data Quality</div>
            </div>
        </section>
        
        <section class="section insights">
            <h3>💡 Key Insights</h3>
            <ul>
                {% for insight in insights %}
                <li>{{ insight }}</li>
                {% endfor %}
            </ul>
        </section>
        
        <section class="section">
            <h2>📈 Executive Summary</h2>
            <div class="chart-container">
                <img src="charts/executive_summary.png" alt="Executive Summary Charts">
            </div>
        </section>
        
        <section class="section">
            <h2>🔤 Product Keywords</h2>
            <div class="chart-container">
                <img src="charts/wordcloud.png" alt="Word Cloud">
            </div>
        </section>
        
        <section class="section">
            <h2>📊 Interactive Visualizations</h2>
            <div class="chart-container">
                <a href="charts/price_distribution_interactive.html" class="chart-link">💰 Price Distribution</a>
                <a href="charts/regional_analysis_interactive.html" class="chart-link">🗺️ Regional Analysis</a>
                <a href="charts/supplier_analysis_interactive.html" class="chart-link">🏭 Supplier Analysis</a>
                <a href="charts/category_sunburst.html" class="chart-link">🌐 Category Hierarchy</a>
                <a href="charts/price_clusters.html" class="chart-link">🎯 Price Segments</a>
            </div>
        </section>
        
        <section class="section">
            <h2>📋 Category Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Products</th>
                        <th>Avg Price</th>
                        <th>Min Price</th>
                        <th>Max Price</th>
                    </tr>
                </thead>
                <tbody>
                    {% for stat in category_stats %}
                    <tr>
                        <td>{{ stat.category }}</td>
                        <td>{{ stat.count }}</td>
                        <td>₹{{ stat.avg_price | default('N/A', true) }}</td>
                        <td>₹{{ stat.min_price | default('N/A', true) }}</td>
                        <td>₹{{ stat.max_price | default('N/A', true) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        
        <footer>
            <p>Generated by Slooze Data Engineering Pipeline | © 2026</p>
        </footer>
    </div>
</body>
</html>
    '''
    
    def __init__(self, df: pd.DataFrame, insights: List[str]):
        self.df = df
        self.insights = insights
        
    def generate(self) -> str:
        """Generate the full HTML report."""
        template = Template(self.REPORT_TEMPLATE)
        
        # Prepare data
        category_stats = db_manager.get_category_stats()
        
        # Format price for display
        avg_price = self.df['price_min'].mean()
        avg_price_str = f"{avg_price/1000:.0f}K" if avg_price < 1000000 else f"{avg_price/100000:.1f}L"
        
        quality_score = (self.df['quality_score'].mean() * 100) if 'quality_score' in self.df.columns else 100
        
        html_content = template.render(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            total_records=len(self.df),
            categories_count=self.df['category'].nunique(),
            suppliers_count=self.df['supplier_name'].nunique(),
            cities_count=self.df['city'].nunique(),
            avg_price=avg_price_str,
            quality_score=f"{quality_score:.0f}",
            insights=self.insights,
            category_stats=category_stats
        )
        
        filepath = REPORTS_DIR / "analysis_report.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filepath)


class AdvancedEDAEngine:
    """
    Main EDA orchestrator.
    
    Combines all analysis components into a cohesive pipeline.
    """
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.insights: List[str] = []
        
    def load_data(self) -> bool:
        """Load data from database or CSV."""
        try:
            # Try database first
            products = db_manager.get_all_products()
            if products:
                self.df = pd.DataFrame(products)
                print(f"[OK] Loaded {len(self.df)} records from database")
                return True
        except Exception as e:
            print(f"[WARN] Database load failed: {e}")
        
        # Fallback to CSV
        csv_path = DATA_DIR / "collected_data.csv"
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
            print(f"[OK] Loaded {len(self.df)} records from CSV")
            return True
        
        print("[ERROR] No data found. Run collector first.")
        return False
    
    def run(self):
        """Execute the full EDA pipeline."""
        from rich.console import Console
        from rich.progress import Progress
        
        console = Console()
        
        console.print("\n[bold magenta]Advanced EDA & Analytics Engine[/bold magenta]")
        console.print("=" * 50)
        
        if not self.load_data():
            return
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Running analysis...", total=6)
            
            # 1. Statistical Analysis
            progress.update(task, description="[cyan]Computing statistics...")
            stats_analyzer = StatisticalAnalyzer(self.df)
            stats_analyzer.compute_summary_stats()
            stats_analyzer.price_distribution_analysis()
            self.insights.extend(stats_analyzer.insights)
            progress.advance(task)
            
            # 2. Generate Visualizations
            progress.update(task, description="[cyan]Creating visualizations...")
            viz_engine = VisualizationEngine(self.df)
            viz_engine.create_price_distribution()
            viz_engine.create_regional_heatmap()
            viz_engine.create_supplier_analysis()
            viz_engine.create_category_sunburst()
            viz_engine.create_static_charts()
            progress.advance(task)
            
            # 3. NLP Analysis
            progress.update(task, description="[cyan]Analyzing text data...")
            nlp_analyzer = NLPAnalyzer(self.df)
            nlp_analyzer.generate_word_cloud()
            keywords = nlp_analyzer.extract_keywords()
            if keywords:
                top_keywords = list(keywords.keys())[:5]
                self.insights.append(f"[Keywords] Top product keywords: {', '.join(top_keywords)}")
            progress.advance(task)
            
            # 4. Clustering
            progress.update(task, description="[cyan]Segmenting products...")
            clustering = ClusteringEngine(self.df)
            self.df = clustering.perform_price_clustering()
            clustering.visualize_clusters()
            
            segment_counts = self.df['price_segment'].value_counts()
            top_segment = segment_counts.index[0] if len(segment_counts) > 0 else "N/A"
            self.insights.append(f"[Segment] Largest market segment: {top_segment} ({segment_counts.iloc[0]} products)")
            progress.advance(task)
            
            # 5. Generate more insights
            progress.update(task, description="[cyan]Generating insights...")
            self._generate_additional_insights()
            progress.advance(task)
            
            # 6. Generate Report
            progress.update(task, description="[cyan]Compiling report...")
            report_gen = ReportGenerator(self.df, self.insights)
            report_path = report_gen.generate()
            progress.advance(task)
        
        console.print("\n[bold green]Analysis Complete![/bold green]")
        console.print(f"\n[bold]Report:[/bold] {report_path}")
        console.print(f"[bold]Charts:[/bold] {CHARTS_DIR}")
        
        console.print("\n[bold yellow]Key Insights:[/bold yellow]")
        for insight in self.insights[:5]:
            console.print(f"  • {insight}")
    
    def _generate_additional_insights(self):
        """Generate additional business insights."""
        # Top city
        top_city = self.df['city'].value_counts().index[0]
        top_city_count = self.df['city'].value_counts().iloc[0]
        self.insights.append(f"[Region] Leading manufacturing hub: {top_city} with {top_city_count} listings")
        
        # Verified suppliers
        verified_pct = (self.df['supplier_verified'].sum() / len(self.df)) * 100
        self.insights.append(f"[Verified] {verified_pct:.1f}% of products are from verified suppliers")
        
        # Price range
        min_price = self.df['price_min'].min()
        max_price = self.df['price_max'].max()
        self.insights.append(f"[Price] Range: Rs {min_price:,.0f} to Rs {max_price:,.0f}")


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/eda.log'),
            logging.StreamHandler()
        ]
    )
    
    engine = AdvancedEDAEngine()
    engine.run()


if __name__ == "__main__":
    main()
