"""
Slooze Data Engineering Challenge - Data Models
================================================

Pydantic models for data validation, serialization, and type safety.
Ensures data quality at ingestion time.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum
import re


class DataSource(str, Enum):
    """Supported data sources."""
    INDIAMART = "indiamart"
    ALIBABA = "alibaba"
    TRADEINDIA = "tradeindia"
    MOCK = "mock_generated"


class PriceUnit(str, Enum):
    """Price unit types."""
    PIECE = "per_piece"
    SET = "per_set"
    KG = "per_kg"
    LOT = "per_lot"
    UNIT = "per_unit"
    UNKNOWN = "unknown"


class ProductListing(BaseModel):
    """
    Validated product listing model.
    
    All scraped data passes through this model for validation
    before being stored in the database.
    """
    
    product_name: str = Field(..., min_length=3, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)
    
    # Pricing
    price_min: Optional[float] = Field(None, ge=0)
    price_max: Optional[float] = Field(None, ge=0)
    price_raw: Optional[str] = None
    price_unit: PriceUnit = PriceUnit.UNKNOWN
    currency: str = Field(default="INR")
    
    # Supplier info
    supplier_name: Optional[str] = Field(None, max_length=300)
    supplier_verified: bool = False
    supplier_rating: Optional[float] = Field(None, ge=0, le=5)
    
    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    country: str = Field(default="India")
    
    # Categorization
    category: str
    subcategory: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Metadata
    source: DataSource
    source_url: Optional[str] = None
    product_id: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.now)
    
    # Quality flags
    is_complete: bool = True
    quality_score: float = Field(default=1.0, ge=0, le=1)

    @field_validator('product_name')
    @classmethod
    def clean_product_name(cls, v: str) -> str:
        """Clean and normalize product name."""
        # Remove extra whitespace
        v = ' '.join(v.split())
        # Remove special characters that might break processing
        v = re.sub(r'[<>{}[\]\\]', '', v)
        return v.strip()

    @field_validator('price_raw')
    @classmethod
    def extract_price_components(cls, v: Optional[str]) -> Optional[str]:
        """Store raw price for reference."""
        return v.strip() if v else None

    @model_validator(mode='after')
    def calculate_quality_score(self):
        """Calculate data quality score based on completeness."""
        score = 0.0
        total_fields = 6
        
        if self.product_name and len(self.product_name) > 10:
            score += 1
        if self.price_min is not None or self.price_max is not None:
            score += 1
        if self.supplier_name:
            score += 1
        if self.city or self.state:
            score += 1
        if self.description:
            score += 1
        if self.category:
            score += 1
            
        self.quality_score = round(score / total_fields, 2)
        self.is_complete = self.quality_score >= 0.7
        
        return self


class SupplierProfile(BaseModel):
    """Supplier aggregated profile."""
    
    supplier_name: str
    total_listings: int = 0
    avg_price: Optional[float] = None
    cities: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    verified: bool = False
    avg_rating: Optional[float] = None


class CategoryStats(BaseModel):
    """Statistics for a product category."""
    
    category: str
    total_products: int
    avg_price: Optional[float]
    min_price: Optional[float]
    max_price: Optional[float]
    top_cities: List[str]
    top_suppliers: List[str]
    price_std: Optional[float] = None


class AnalysisResult(BaseModel):
    """Container for analysis results."""
    
    generated_at: datetime = Field(default_factory=datetime.now)
    total_records: int
    categories_analyzed: List[str]
    
    # Summary statistics
    overall_avg_price: Optional[float]
    overall_median_price: Optional[float]
    price_range: tuple
    
    # Quality metrics
    avg_quality_score: float
    complete_records_pct: float
    
    # Insights
    insights: List[str] = Field(default_factory=list)
