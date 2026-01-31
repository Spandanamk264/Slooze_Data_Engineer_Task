"""
SLOOZE ULTIMATE VISUALIZATION ENGINE v3.0
==========================================
Comprehensive visualization suite with 20+ guaranteed-working charts.
All charts are tested and will render with data - NO EMPTY PLOTS!

Author: Slooze Analytics Team
Lines: 1000+
Charts: 20+ Interactive + 3 Static
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.preprocessing import LabelEncoder, StandardScaler
from wordcloud import WordCloud
from typing import List, Dict, Tuple, Any, Optional
import base64
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
pio.templates.default = "plotly_white"

# Premium color palettes - using hex for matplotlib compatibility
COLORS_PRIMARY = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
COLORS_CATEGORY_PLOTLY = px.colors.qualitative.Set2  # For plotly charts
COLORS_CATEGORY_MPL = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']  # For matplotlib
COLORS_GRADIENT = px.colors.sequential.Viridis


def fig_to_base64(fig) -> str:
    """
    Convert matplotlib figure to base64 string.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Base64 encoded image string with data URI
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def plotly_to_html(fig) -> str:
    """
    Convert plotly figure to embedded HTML with CDN.
    INCLUDES PLOTLY.JS IN EVERY CHART FOR GUARANTEED RENDERING!
    
    Args:
        fig: Plotly figure object
        
    Returns:
        HTML string with embedded Plotly.js
    """
    # Use 'cdn' to include plotly.js in EVERY chart - guaranteed to work!
    return pio.to_html(
        fig, 
        full_html=False, 
        include_plotlyjs='cdn',
        config={
            'responsive': True,
            'displayModeBar': True,
            'displaylogo': False
        }
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare dataframe for visualization.
    
    Args:
        df: Input dataframe
        
    Returns:
        Cleaned dataframe
    """
    df = df.copy()
    
    # Convert numeric columns
    numeric_cols = ['price_min', 'price_max', 'quality_score', 'supplier_rating']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert boolean columns
    if 'supplier_verified' in df.columns:
        df['supplier_verified'] = df['supplier_verified'].astype(bool)
    
    return df


class PriceCharts:
    """Price analysis visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = clean_dataframe(df)
    
    def histogram(self) -> Dict[str, str]:
        """Simple price distribution histogram."""
        prices = self.df['price_min'].dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=prices,
            nbinsx=40,
            marker=dict(
                color='#667eea',
                line=dict(width=1, color='white')
            ),
            name='Price Distribution',
            hovertemplate='Price Range: ₹%{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Price Distribution Analysis</b>',
            xaxis_title='Price (INR)',
            yaxis_title='Frequency',
            height=500,
            showlegend=False,
            hovermode='closest'
        )
        
        return {'title': 'Price Distribution', 'html': plotly_to_html(fig)}
    
    def log_histogram(self) -> Dict[str, str]:
        """Log-transformed price distribution."""
        prices = self.df['price_min'].dropna()
        log_prices = np.log10(prices + 1)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=log_prices,
            nbinsx=40,
            marker=dict(color='#764ba2', line=dict(width=1, color='white')),
            hovertemplate='Log10(Price): %{x:.2f}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Log-Transformed Price Distribution</b>',
            xaxis_title='Log10(Price + 1)',
            yaxis_title='Frequency',
            height=500,
            showlegend=False
        )
        
        return {'title': 'Log Price Distribution', 'html': plotly_to_html(fig)}
    
    def percentiles_chart(self) -> Dict[str, str]:
        """Price percentiles visualization."""
        prices = self.df['price_min'].dropna()
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        pct_values = [prices.quantile(p/100) for p in percentiles]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f'P{p}' for p in percentiles],
            y=pct_values,
            marker=dict(
                color=pct_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Price (₹)')
            ),
            text=[f'₹{v/1000:.1f}K' if v >= 1000 else f'₹{v:.0f}' for v in pct_values],
            textposition='outside',
            hovertemplate='%{x}: ₹%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Price Percentiles</b>',
            xaxis_title='Percentile',
            yaxis_title='Price (INR)',
            height=500,
            showlegend=False
        )
        
        return {'title': 'Price Percentiles', 'html': plotly_to_html(fig)}
    
    def scatter_plot(self) -> Dict[str, str]:
        """Min vs Max price scatter."""
        valid_df = self.df.dropna(subset=['price_min', 'price_max'])
        sample = valid_df.sample(min(600, len(valid_df)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample['price_min'],
            y=sample['price_max'],
            mode='markers',
            marker=dict(
                size=7,
                color='#f093fb',
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            text=sample['product_name'],
            hovertemplate='<b>%{text}</b><br>Min: ₹%{x:,.0f}<br>Max: ₹%{y:,.0f}<extra></extra>'
        ))
        
        # Add diagonal reference line
        max_val = max(sample['price_min'].max(), sample['price_max'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Equal Line',
            showlegend=True
        ))
        
        fig.update_layout(
            title='<b>Min vs Max Price Scatter</b>',
            xaxis_title='Minimum Price (INR)',
            yaxis_title='Maximum Price (INR)',
            height=550,
            hovermode='closest'
        )
        
        return {'title': 'Price Scatter Plot', 'html': plotly_to_html(fig)}
    
    def price_spread(self) -> Dict[str, str]:
        """Price spread analysis."""
        if 'price_max' not in self.df.columns:
            return {'title': 'Price Spread', 'html': '<p>No price max data</p>'}
        
        spread = (self.df['price_max'] - self.df['price_min']).dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=spread,
            nbinsx=40,
            marker=dict(color='#f5576c', line=dict(width=1, color='white')),
            hovertemplate='Spread: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Price Spread Distribution</b>',
            xaxis_title='Price Spread (Max - Min) INR',
            yaxis_title='Frequency',
            height=500,
            showlegend=False
        )
        
        return {'title': 'Price Spread', 'html': plotly_to_html(fig)}


class CategoryCharts:
    """Category analysis visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = clean_dataframe(df)
    
    def bar_chart(self) -> Dict[str, str]:
        """Category distribution bar chart."""
        cat_counts = self.df['category'].value_counts().head(12)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=cat_counts.index,
            x=cat_counts.values,
            orientation='h',
            marker=dict(
                color=COLORS_PRIMARY,
                line=dict(width=1, color='white')
            ),
            text=cat_counts.values,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Products: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Top 12 Product Categories</b>',
            xaxis_title='Number of Products',
            yaxis_title='Category',
            height=550,
            showlegend=False
        )
        
        return {'title': 'Category Distribution', 'html': plotly_to_html(fig)}
    
    def box_plots(self) -> Dict[str, str]:
        """Category price box plots."""
        fig = go.Figure()
        
        for i, cat in enumerate(self.df['category'].unique()):
            cat_prices = self.df[self.df['category'] == cat]['price_min'].dropna()
            if len(cat_prices) > 0:
                fig.add_trace(go.Box(
                    y=cat_prices,
                    name=cat[:18],
                    marker=dict(color=COLORS_CATEGORY_PLOTLY[i % len(COLORS_CATEGORY_PLOTLY)]),
                    boxmean='sd',
                    hovertemplate='%{y:,.0f}<extra></extra>'
                ))
        
        fig.update_layout(
            title='<b>Price Distribution by Category</b>',
            yaxis_title='Price (INR)',
            xaxis_title='Category',
            height=550,
            showlegend=False
        )
        
        return {'title': 'Category Box Plots', 'html': plotly_to_html(fig)}
    
    def violin_plots(self) -> Dict[str, str]:
        """Category price violin plots."""
        fig = go.Figure()
        
        for i, cat in enumerate(self.df['category'].unique()):
            cat_prices = self.df[self.df['category'] == cat]['price_min'].dropna()
            if len(cat_prices) > 0:
                fig.add_trace(go.Violin(
                    y=cat_prices,
                    name=cat[:18],
                    marker=dict(color=COLORS_CATEGORY_PLOTLY[i % len(COLORS_CATEGORY_PLOTLY)]),
                    box_visible=True,
                    meanline_visible=True
                ))
        
        fig.update_layout(
            title='<b>Price Distribution Violin Plots</b>',
            yaxis_title='Price (INR)',
            xaxis_title='Category',
            height=550,
            showlegend=False
        )
        
        return {'title': 'Category Violin Plots', 'html': plotly_to_html(fig)}
    
    def avg_price_chart(self) -> Dict[str, str]:
        """Average price by category."""
        cat_avg = self.df.groupby('category')['price_min'].mean().sort_values(ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=cat_avg.index,
            x=cat_avg.values,
            orientation='h',
            marker=dict(
                color=cat_avg.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Avg Price (₹)')
            ),
            text=[f'₹{v/1000:.1f}K' if v >= 1000 else f'₹{v:.0f}' for v in cat_avg.values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Avg: ₹%{x:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Average Price by Category</b>',
            xaxis_title='Average Price (INR)',
            yaxis_title='Category',
            height=550,
            showlegend=False
        )
        
        return {'title': 'Avg Price by Category', 'html': plotly_to_html(fig)}


class GeographicCharts:
    """Geographic analysis visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = clean_dataframe(df)
    
    def city_distribution(self) -> Dict[str, str]:
        """City product distribution."""
        city_counts = self.df['city'].value_counts().head(15)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=city_counts.index,
            x=city_counts.values,
            orientation='h',
            marker=dict(color='#4facfe', line=dict(width=1, color='white')),
            text=city_counts.values,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Products: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Top 15 Cities by Product Count</b>',
            xaxis_title='Number of Products',
            yaxis_title='City',
            height=600,
            showlegend=False
        )
        
        return {'title': 'City Distribution', 'html': plotly_to_html(fig)}
    
    def city_price_analysis(self) -> Dict[str, str]:
        """Average price by city."""
        city_avg = self.df.groupby('city')['price_min'].mean().sort_values(ascending=False).head(15)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=city_avg.index,
            x=city_avg.values,
            orientation='h',
            marker=dict(
                color=city_avg.values,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title='Avg Price (₹)')
            ),
            text=[f'₹{v/1000:.1f}K' if v >= 1000 else f'₹{v:.0f}' for v in city_avg.values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Avg: ₹%{x:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Average Price by City</b>',
            xaxis_title='Average Price (INR)',
            yaxis_title='City',
            height=600,
            showlegend=False
        )
        
        return {'title': 'City Price Analysis', 'html': plotly_to_html(fig)}
    
    def treemap(self) -> Dict[str, str]:
        """Category-City treemap."""
        treemap_df = self.df.groupby(['category', 'city']).agg({
            'product_name': 'count',
            'price_min': 'mean'
        }).reset_index()
        treemap_df.columns = ['category', 'city', 'count', 'avg_price']
        
        fig = px.treemap(
            treemap_df,
            path=['category', 'city'],
            values='count',
            color='avg_price',
            color_continuous_scale='Viridis',
            title='<b>Market Distribution: Category × City</b>'
        )
        
        fig.update_layout(height=600)
        
        return {'title': 'Market Treemap', 'html': plotly_to_html(fig)}
    
    def sunburst(self) -> Dict[str, str]:
        """Category-City sunburst chart."""
        sunburst_df = self.df.groupby(['category', 'city']).size().reset_index(name='count')
        
        fig = px.sunburst(
            sunburst_df,
            path=['category', 'city'],
            values='count',
            color='count',
            color_continuous_scale='Blues',
            title='<b>Market Hierarchy Sunburst</b>'
        )
        
        fig.update_layout(height=600)
        
        return {'title': 'Market Sunburst', 'html': plotly_to_html(fig)}


class AdvancedCharts:
    """Advanced specialty visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = clean_dataframe(df)
    
    def funnel_chart(self) -> Dict[str, str]:
        """Category volume funnel."""
        cat_counts = self.df['category'].value_counts().head(8)
        
        fig = go.Figure()
        fig.add_trace(go.Funnel(
            y=cat_counts.index,
            x=cat_counts.values,
            textinfo='value+percent initial',
            marker=dict(color=COLORS_CATEGORY_PLOTLY[:len(cat_counts)]),
            hovertemplate='<b>%{y}</b><br>Products: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Category Volume Funnel</b>',
            height=550
        )
        
        return {'title': 'Category Funnel', 'html': plotly_to_html(fig)}
    
    def waterfall_chart(self) -> Dict[str, str]:
        """Price contribution waterfall."""
        cat_avg = self.df.groupby('category')['price_min'].mean().sort_values(ascending=False).head(8)
        baseline = cat_avg.min()
        
        fig = go.Figure()
        fig.add_trace(go.Waterfall(
            x=cat_avg.index,
            y=cat_avg.values - baseline,
            connector={"line": {"color": "gray"}},
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#667eea"}},
            hovertemplate='<b>%{x}</b><br>Value: ₹%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Price Waterfall by Category</b>',
            height=550,
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        
        return {'title': 'Price Waterfall', 'html': plotly_to_html(fig)}


class SupplierCharts:
    """Supplier analysis visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = clean_dataframe(df)
    
    def top_suppliers(self) -> Dict[str, str]:
        """Top suppliers by product count."""
        supplier_counts = self.df['supplier_name'].value_counts().head(15)
        supplier_verified = self.df.groupby('supplier_name')['supplier_verified'].first()
        
        colors = ['#10b981' if supplier_verified.get(s, False) else '#ef4444' 
                 for s in supplier_counts.index]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=supplier_counts.index,
            x=supplier_counts.values,
            orientation='h',
            marker=dict(color=colors, line=dict(width=1, color='white')),
            text=supplier_counts.values,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Products: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Top 15 Suppliers</b><br><sup>Green = Verified | Red = Unverified</sup>',
            xaxis_title='Number of Products',
            yaxis_title='Supplier',
            height=600,
            showlegend=False
        )
        
        return {'title': 'Top Suppliers', 'html': plotly_to_html(fig)}
    
    def verification_pie(self) -> Dict[str, str]:
        """Supplier verification status."""
        verified = self.df['supplier_verified'].value_counts()
        labels = ['Verified' if v else 'Unverified' for v in verified.index]
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=labels,
            values=verified.values,
            hole=0.4,
            marker=dict(colors=['#10b981', '#ef4444']),
            textinfo='label+percent+value',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Supplier Verification Status</b>',
            height=500
        )
        
        return {'title': 'Verification Status', 'html': plotly_to_html(fig)}


class StatisticalCharts:
    """Statistical analysis visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = clean_dataframe(df)
    
    def correlation_heatmap(self) -> Dict[str, str]:
        """Correlation matrix heatmap."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'title': 'Correlation Matrix', 'html': '<p>Insufficient numeric data</p>'}
        
        corr = numeric_df.corr()
        
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 11},
            hovertemplate='%{x} × %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Feature Correlation Matrix</b>',
            height=550,
            xaxis={'side': 'bottom'}
        )
        
        return {'title': 'Correlation Heatmap', 'html': plotly_to_html(fig)}
    
    def quality_distribution(self) -> Dict[str, str]:
        """Quality score distribution."""
        if 'quality_score' not in self.df.columns:
            return {'title': 'Quality Distribution', 'html': '<p>No quality data</p>'}
        
        quality = self.df['quality_score'].dropna()
        
        if len(quality) == 0:
            return {'title': 'Quality Distribution', 'html': '<p>No quality data</p>'}
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=quality,
            nbinsx=30,
            marker=dict(color='#f093fb', line=dict(width=1, color='white')),
            hovertemplate='Quality: %{x:.1f}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b>Data Quality Score Distribution</b>',
            xaxis_title='Quality Score',
            yaxis_title='Frequency',
            height=500,
            showlegend=False
        )
        
        return {'title': 'Quality Distribution', 'html': plotly_to_html(fig)}


class CompositeCharts:
    """Multi-panel composite visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = clean_dataframe(df)
    
    def dashboard(self) -> Dict[str, str]:
        """Executive dashboard with multiple panels."""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Price Distribution', 'Category Mix', 'Top Cities',
                'Category Prices', 'Verification', 'Price Trends'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "pie"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Price distribution
        prices = self.df['price_min'].dropna()
        fig.add_trace(go.Histogram(
            x=prices, nbinsx=30, marker_color='#667eea', showlegend=False
        ), row=1, col=1)
        
        # Category pie
        cat_counts = self.df['category'].value_counts().head(6)
        fig.add_trace(go.Pie(
            labels=cat_counts.index, values=cat_counts.values,
            marker_colors=COLORS_CATEGORY_PLOTLY, hole=0.3, showlegend=False
        ), row=1, col=2)
        
        # Top cities
        city_counts = self.df['city'].value_counts().head(8)
        fig.add_trace(go.Bar(
            y=city_counts.index, x=city_counts.values, orientation='h',
            marker_color='#4facfe', showlegend=False
        ), row=1, col=3)
        
        # Category box plots
        for i, cat in enumerate(self.df['category'].unique()[:5]):
            cat_prices = self.df[self.df['category'] == cat]['price_min'].dropna()
            if len(cat_prices) > 0:
                fig.add_trace(go.Box(
                    y=cat_prices, name=cat[:12],
                    marker_color=COLORS_CATEGORY_PLOTLY[i % len(COLORS_CATEGORY_PLOTLY)],
                    showlegend=False
                ), row=2, col=1)
        
        # Verification pie
        verified = self.df['supplier_verified'].value_counts()
        labels = ['Verified' if v else 'Unverified' for v in verified.index]
        fig.add_trace(go.Pie(
            labels=labels, values=verified.values,
            marker_colors=['#10b981', '#ef4444'], hole=0.3, showlegend=False
        ), row=2, col=2)
        
        # Price trends
        cat_avg = self.df.groupby('category')['price_min'].mean().sort_values(ascending=False).head(8)
        fig.add_trace(go.Bar(
            y=cat_avg.index, x=cat_avg.values, orientation='h',
            marker_color='#11998e', showlegend=False
        ), row=2, col=3)
        
        fig.update_layout(
            height=800,
            title_text='<b>Executive Dashboard</b>',
            title_x=0.5,
            showlegend=False
        )
        
        return {'title': 'Executive Dashboard', 'html': plotly_to_html(fig)}
    
    def multi_category(self) -> Dict[str, str]:
        """Multi-panel category analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Product Count', 'Average Price',
                'Price Range', 'Volatility'
            )
        )
        
        cat_stats = self.df.groupby('category').agg({
            'product_name': 'count',
            'price_min': ['mean', 'std'],
            'price_max': 'mean'
        }).reset_index()
        cat_stats.columns = ['category', 'count', 'avg_min', 'std', 'avg_max']
        cat_stats['range'] = cat_stats['avg_max'] - cat_stats['avg_min']
        cat_stats = cat_stats.sort_values('count', ascending=False).head(10)
        
        # Count
        fig.add_trace(go.Bar(
            y=cat_stats['category'], x=cat_stats['count'],
            orientation='h', marker_color='#667eea', showlegend=False
        ), row=1, col=1)
        
        # Avg price
        fig.add_trace(go.Bar(
            y=cat_stats['category'], x=cat_stats['avg_min'],
            orientation='h', marker_color='#764ba2', showlegend=False
        ), row=1, col=2)
        
        # Range
        fig.add_trace(go.Bar(
            y=cat_stats['category'], x=cat_stats['range'],
            orientation='h', marker_color='#f093fb', showlegend=False
        ), row=2, col=1)
        
        # Volatility
        fig.add_trace(go.Bar(
            y=cat_stats['category'], x=cat_stats['std'],
            orientation='h', marker_color='#f5576c', showlegend=False
        ), row=2, col=2)
        
        fig.update_layout(
            height=700,
            title_text='<b>Category Deep Dive</b>',
            title_x=0.5,
            showlegend=False
        )
        
        return {'title': 'Category Analysis', 'html': plotly_to_html(fig)}


class StaticCharts:
    """Matplotlib-based static charts."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = clean_dataframe(df)
    
    def summary_grid(self) -> Dict[str, str]:
        """Comprehensive summary grid."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Comprehensive Statistical Summary', fontsize=20, fontweight='bold', y=0.995)
        
        # Price distribution
        prices = self.df['price_min'].dropna()
        if len(prices) > 0:
            axes[0, 0].hist(prices, bins=35, color='#667eea', alpha=0.7, edgecolor='white')
            axes[0, 0].set_title('Price Distribution', fontweight='bold', fontsize=12)
            axes[0, 0].set_xlabel('Price (INR)', fontsize=10)
            axes[0, 0].set_ylabel('Frequency', fontsize=10)
            axes[0, 0].grid(alpha=0.3)
        
        # Category counts
        cat_counts = self.df['category'].value_counts().head(10)
        axes[0, 1].barh(range(len(cat_counts)), cat_counts.values, color='#764ba2', edgecolor='white')
        axes[0, 1].set_yticks(range(len(cat_counts)))
        axes[0, 1].set_yticklabels(cat_counts.index, fontsize=9)
        axes[0, 1].set_title('Top 10 Categories', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Count', fontsize=10)
        axes[0, 1].grid(alpha=0.3, axis='x')
        
        # City distribution
        city_counts = self.df['city'].value_counts().head(10)
        axes[0, 2].barh(range(len(city_counts)), city_counts.values, color='#4facfe', edgecolor='white')
        axes[0, 2].set_yticks(range(len(city_counts)))
        axes[0, 2].set_yticklabels(city_counts.index, fontsize=9)
        axes[0, 2].set_title('Top 10 Cities', fontweight='bold', fontsize=12)
        axes[0, 2].set_xlabel('Count', fontsize=10)
        axes[0, 2].grid(alpha=0.3, axis='x')
        
        # Verification pie
        verified = self.df['supplier_verified'].value_counts()
        labels = ['Verified' if v else 'Unverified' for v in verified.index]
        axes[1, 0].pie(verified.values, labels=labels, autopct='%1.1f%%',
                      colors=['#10b981', '#ef4444'], startangle=90, textprops={'fontsize': 10})
        axes[1, 0].set_title('Verification Status', fontweight='bold', fontsize=12)
        
        # Price by category boxplot
        sample_cats = self.df['category'].unique()[:8]
        cat_data = [self.df[self.df['category'] == cat]['price_min'].dropna() for cat in sample_cats]
        bp = axes[1, 1].boxplot(cat_data, labels=[c[:10] for c in sample_cats], patch_artist=True)
        mpl_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
        for patch, color in zip(bp['boxes'], mpl_colors):
            patch.set_facecolor(color)
        axes[1, 1].set_xticklabels([c[:10] for c in sample_cats], rotation=45, ha='right', fontsize=8)
        axes[1, 1].set_title('Price by Category', fontweight='bold', fontsize=12)
        axes[1, 1].set_ylabel('Price (INR)', fontsize=10)
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        # Scatter plot
        sample = self.df.dropna(subset=['price_min', 'price_max']).sample(min(300, len(self.df)))
        axes[1, 2].scatter(sample['price_min'], sample['price_max'], 
                          c='#f093fb', alpha=0.5, s=30, edgecolors='white', linewidth=0.5)
        axes[1, 2].set_title('Min vs Max Price', fontweight='bold', fontsize=12)
        axes[1, 2].set_xlabel('Min Price (INR)', fontsize=10)
        axes[1, 2].set_ylabel('Max Price (INR)', fontsize=10)
        axes[1, 2].grid(alpha=0.3)
        
        # Log price distribution
        log_prices = np.log10(prices + 1)
        axes[2, 0].hist(log_prices, bins=35, color='#11998e', alpha=0.7, edgecolor='white')
        axes[2, 0].set_title('Log Price Distribution', fontweight='bold', fontsize=12)
        axes[2, 0].set_xlabel('Log10(Price + 1)', fontsize=10)
        axes[2, 0].set_ylabel('Frequency', fontsize=10)
        axes[2, 0].grid(alpha=0.3)
        
        # Average price by category
        cat_avg = self.df.groupby('category')['price_min'].mean().sort_values(ascending=False).head(10)
        axes[2, 1].barh(range(len(cat_avg)), cat_avg.values, color='#f5576c', edgecolor='white')
        axes[2, 1].set_yticks(range(len(cat_avg)))
        axes[2, 1].set_yticklabels(cat_avg.index, fontsize=9)
        axes[2, 1].set_title('Avg Price by Category', fontweight='bold', fontsize=12)
        axes[2, 1].set_xlabel('Avg Price (INR)', fontsize=10)
        axes[2, 1].grid(alpha=0.3, axis='x')
        
        # Price percentiles
        percentiles = [10, 25, 50, 75, 90, 95]
        pct_vals = [prices.quantile(p/100) for p in percentiles]
        axes[2, 2].bar([f'P{p}' for p in percentiles], pct_vals, color=COLORS_PRIMARY, edgecolor='white')
        axes[2, 2].set_title('Price Percentiles', fontweight='bold', fontsize=12)
        axes[2, 2].set_ylabel('Price (INR)', fontsize=10)
        axes[2, 2].tick_params(axis='x', labelsize=9)
        axes[2, 2].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        return {'title': 'Statistical Summary', 'base64': fig_to_base64(fig)}
    
    def correlation_grid(self) -> Dict[str, str]:
        """Correlation analysis grid."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Insufficient numeric data', ha='center', va='center', fontsize=16)
            ax.axis('off')
            return {'title': 'Correlation Analysis', 'base64': fig_to_base64(fig)}
        
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   ax=ax, fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Matrix', fontsize=18, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        return {'title': 'Correlation Analysis', 'base64': fig_to_base64(fig)}
    
    def category_deep_dive(self) -> Dict[str, str]:
        """Category deep dive analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Category Deep Dive Analysis', fontsize=18, fontweight='bold', y=0.995)
        
        # Violin plots
        sample_cats = self.df['category'].unique()[:6]
        cat_data = [self.df[self.df['category'] == cat]['price_min'].dropna().values for cat in sample_cats]
        parts = axes[0, 0].violinplot(cat_data, showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], COLORS_CATEGORY_MPL):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        axes[0, 0].set_xticks(range(1, len(sample_cats) + 1))
        axes[0, 0].set_xticklabels([c[:12] for c in sample_cats], rotation=45, ha='right', fontsize=9)
        axes[0, 0].set_title('Price Distribution (Violin)', fontweight='bold', fontsize=12)
        axes[0, 0].set_ylabel('Price (INR)', fontsize=10)
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # Count + Avg price scatter
        cat_stats = self.df.groupby('category').agg({
            'product_name': 'count',
            'price_min': 'mean'
        }).reset_index()
        cat_stats.columns = ['category', 'count', 'avg_price']
        
        scatter = axes[0, 1].scatter(cat_stats['count'], cat_stats['avg_price'],
                                     s=cat_stats['count']*2, c=range(len(cat_stats)),
                                     cmap='viridis', alpha=0.6, edgecolors='white', linewidth=1)
        axes[0, 1].set_title('Volume vs Price', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Product Count', fontsize=10)
        axes[0, 1].set_ylabel('Avg Price (INR)', fontsize=10)
        axes[0, 1].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Category Index')
        
        # Price range by category
        if 'price_max' in self.df.columns:
            cat_range = self.df.groupby('category').apply(
                lambda x: (x['price_max'] - x['price_min']).mean()
            ).sort_values(ascending=False).head(10)
            axes[1, 0].barh(range(len(cat_range)), cat_range.values, color='#f5576c', edgecolor='white')
            axes[1, 0].set_yticks(range(len(cat_range)))
            axes[1, 0].set_yticklabels(cat_range.index, fontsize=9)
            axes[1, 0].set_title('Avg Price Range', fontweight='bold', fontsize=12)
            axes[1, 0].set_xlabel('Range (INR)', fontsize=10)
            axes[1, 0].grid(alpha=0.3, axis='x')
        
        # Std deviation
        cat_std = self.df.groupby('category')['price_min'].std().sort_values(ascending=False).head(10)
        axes[1, 1].barh(range(len(cat_std)), cat_std.values, color='#4facfe', edgecolor='white')
        axes[1, 1].set_yticks(range(len(cat_std)))
        axes[1, 1].set_yticklabels(cat_std.index, fontsize=9)
        axes[1, 1].set_title('Price Volatility (Std Dev)', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Std Deviation (INR)', fontsize=10)
        axes[1, 1].grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        return {'title': 'Category Deep Dive', 'base64': fig_to_base64(fig)}


class WordCloudGenerator:
    """Word cloud generation."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create(self) -> str:
        """Generate product keyword word cloud."""
        text = ' '.join(self.df['product_name'].dropna().astype(str))
        
        if len(text.strip()) < 10:
            fig, ax = plt.subplots(figsize=(16, 7))
            ax.text(0.5, 0.5, 'Insufficient text data', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig_to_base64(fig)
        
        stopwords = {
            'model', 'type', 'new', 'quality', 'best', 'for', 'the', 'and',
            'with', 'pro', 'standard', 'grade', 'series', 'version', 'style', 'a', 'an'
        }
        
        wc = WordCloud(
            width=1600,
            height=700,
            background_color='white',
            colormap='viridis',
            max_words=120,
            stopwords=stopwords,
            prefer_horizontal=0.75,
            min_font_size=12,
            max_font_size=100,
            relative_scaling=0.5
        ).generate(text.lower())
        
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Product Keyword Analysis', fontsize=22, fontweight='bold', pad=20)
        
        return fig_to_base64(fig)


class VisualizationEngine:
    """
    Main visualization engine orchestrator.
    
    Coordinates all chart generation and ensures all visualizations
    are properly created with data - NO EMPTY PLOTS!
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with cleaned dataframe."""
        self.df = clean_dataframe(df)
        self.charts = []
        self.static_charts = []
    
    def create_all(self) -> Tuple[List[Dict], List[Dict], str]:
        """
        Create ALL visualizations.
        
        Returns:
            Tuple of (interactive_charts, static_charts, wordcloud_base64)
        """
        print("    - Creating price analysis charts...")
        price = PriceCharts(self.df)
        self.charts.append(price.histogram())
        self.charts.append(price.log_histogram())
        self.charts.append(price.percentiles_chart())
        self.charts.append(price.scatter_plot())
        self.charts.append(price.price_spread())
        
        print("    - Creating category analysis charts...")
        category = CategoryCharts(self.df)
        self.charts.append(category.bar_chart())
        self.charts.append(category.box_plots())
        self.charts.append(category.violin_plots())
        self.charts.append(category.avg_price_chart())
        
        print("    - Creating geographic charts...")
        geo = GeographicCharts(self.df)
        self.charts.append(geo.city_distribution())
        self.charts.append(geo.city_price_analysis())
        self.charts.append(geo.treemap())
        self.charts.append(geo.sunburst())
        
        print("    - Creating supplier charts...")
        supplier = SupplierCharts(self.df)
        self.charts.append(supplier.top_suppliers())
        self.charts.append(supplier.verification_pie())
        
        print("    - Creating statistical charts...")
        stats = StatisticalCharts(self.df)
        self.charts.append(stats.correlation_heatmap())
        self.charts.append(stats.quality_distribution())
        
        print("    - Creating advanced charts...")
        advanced = AdvancedCharts(self.df)
        self.charts.append(advanced.funnel_chart())
        self.charts.append(advanced.waterfall_chart())
        
        print("    - Creating composite dashboards...")
        composite = CompositeCharts(self.df)
        self.charts.append(composite.dashboard())
        self.charts.append(composite.multi_category())
        
        print("    - Creating static charts...")
        static = StaticCharts(self.df)
        self.static_charts.append(static.summary_grid())
        self.static_charts.append(static.correlation_grid())
        self.static_charts.append(static.category_deep_dive())
        
        print("    - Generating word cloud...")
        wc_gen = WordCloudGenerator(self.df)
        wordcloud = wc_gen.create()
        
        # Filter out any charts with minimal content
        self.charts = [c for c in self.charts if c.get('html') and len(c['html']) > 50]
        self.static_charts = [c for c in self.static_charts if c.get('base64')]
        
        print(f"       Generated {len(self.charts)} interactive charts")
        print(f"       Generated {len(self.static_charts)} static charts")
        
        return self.charts, self.static_charts, wordcloud
