"""
Slooze Data Engineering Challenge - Configuration Module
=========================================================

Centralized configuration for the entire data pipeline.
Supports environment variables for production deployments.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, OUTPUT_DIR, LOGS_DIR, OUTPUT_DIR / "charts", OUTPUT_DIR / "reports"]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ScraperConfig:
    """Configuration for web scraping operations."""
    
    # Target marketplaces
    SOURCES: Dict[str, str] = field(default_factory=lambda: {
        "indiamart": "https://dir.indiamart.com",
        "alibaba": "https://www.alibaba.com",
        "tradeindia": "https://www.tradeindia.com"
    })
    
    # Target product categories
    CATEGORIES: List[str] = field(default_factory=lambda: [
        "industrial-pumps",
        "electric-motors", 
        "cnc-machines",
        "textile-machinery",
        "packaging-machines"
    ])
    
    # Request settings
    REQUEST_TIMEOUT: int = 15
    MAX_RETRIES: int = 5
    RETRY_BACKOFF: float = 2.0
    
    # Rate limiting (requests per minute)
    RATE_LIMIT: int = 30
    
    # Concurrency
    MAX_CONCURRENT_REQUESTS: int = 5
    
    # User agent rotation
    ROTATE_USER_AGENTS: bool = True
    
    # Proxy support (optional)
    USE_PROXIES: bool = False
    PROXY_LIST: List[str] = field(default_factory=list)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    DB_PATH: Path = DATA_DIR / "slooze_data.db"
    ECHO_SQL: bool = False
    
    @property
    def connection_string(self) -> str:
        return f"sqlite:///{self.DB_PATH}"


@dataclass  
class AnalysisConfig:
    """Configuration for EDA and analysis."""
    
    # Visualization settings
    CHART_STYLE: str = "seaborn-v0_8-darkgrid"
    COLOR_PALETTE: str = "viridis"
    FIGURE_DPI: int = 150
    
    # NLP settings
    MIN_WORD_FREQUENCY: int = 3
    MAX_FEATURES: int = 1000
    
    # Clustering
    N_CLUSTERS: int = 5
    
    # Report generation
    REPORT_TEMPLATE: str = "premium"


# Global config instances
scraper_config = ScraperConfig()
db_config = DatabaseConfig()
analysis_config = AnalysisConfig()
