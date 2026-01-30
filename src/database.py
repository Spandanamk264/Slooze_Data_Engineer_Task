"""
Slooze Data Engineering Challenge - Database Layer
===================================================

SQLAlchemy ORM models and database operations.
Provides persistent storage with full CRUD operations.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    DateTime, Text, Enum as SQLEnum, Index, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from config import db_config

logger = logging.getLogger(__name__)

Base = declarative_base()


class ProductTable(Base):
    """SQLAlchemy model for product listings."""
    
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core fields
    product_name = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    
    # Pricing
    price_min = Column(Float)
    price_max = Column(Float)
    price_raw = Column(String(100))
    price_unit = Column(String(50))
    currency = Column(String(10), default='INR')
    
    # Supplier
    supplier_name = Column(String(300), index=True)
    supplier_verified = Column(Boolean, default=False)
    supplier_rating = Column(Float)
    
    # Location
    city = Column(String(100), index=True)
    state = Column(String(100))
    country = Column(String(100), default='India')
    
    # Categorization
    category = Column(String(100), nullable=False, index=True)
    subcategory = Column(String(100))
    tags = Column(Text)  # Stored as JSON string
    
    # Metadata
    source = Column(String(50), nullable=False)
    source_url = Column(String(1000))
    product_id = Column(String(100))
    scraped_at = Column(DateTime, default=datetime.now)
    
    # Quality
    is_complete = Column(Boolean, default=True)
    quality_score = Column(Float, default=1.0)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_category_city', 'category', 'city'),
        Index('idx_price_range', 'price_min', 'price_max'),
        Index('idx_source_date', 'source', 'scraped_at'),
    )


class ScrapingRunTable(Base):
    """Track scraping runs for auditing."""
    
    __tablename__ = 'scraping_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime)
    
    source = Column(String(50))
    categories_scraped = Column(Text)
    
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    records_collected = Column(Integer, default=0)
    records_validated = Column(Integer, default=0)
    records_rejected = Column(Integer, default=0)
    
    status = Column(String(20), default='running')
    error_message = Column(Text)


class DatabaseManager:
    """
    Manages database connections and operations.
    
    Features:
    - Connection pooling
    - Transaction management
    - Batch operations for performance
    """
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or db_config.connection_string
        self.engine = create_engine(
            self.connection_string,
            echo=db_config.ECHO_SQL,
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._create_tables()
        
    def _create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at {self.connection_string}")
    
    def get_session(self):
        """Get a new database session."""
        return self.SessionLocal()
    
    def insert_products(self, products: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Batch insert products for performance.
        
        Args:
            products: List of product dictionaries
            batch_size: Number of records per batch
            
        Returns:
            Number of records inserted
        """
        session = self.get_session()
        inserted = 0
        
        try:
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                for product_data in batch:
                    # Convert tags list to string
                    if 'tags' in product_data and isinstance(product_data['tags'], list):
                        product_data['tags'] = ','.join(product_data['tags'])
                    
                    product = ProductTable(**product_data)
                    session.add(product)
                    inserted += 1
                
                session.commit()
                logger.debug(f"Inserted batch of {len(batch)} products")
                
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert products: {e}")
            raise
        finally:
            session.close()
            
        logger.info(f"Inserted {inserted} products into database")
        return inserted
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        """Retrieve all products as dictionaries."""
        session = self.get_session()
        try:
            products = session.query(ProductTable).all()
            return [self._product_to_dict(p) for p in products]
        finally:
            session.close()
    
    def get_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get products filtered by category."""
        session = self.get_session()
        try:
            products = session.query(ProductTable).filter(
                ProductTable.category == category
            ).all()
            return [self._product_to_dict(p) for p in products]
        finally:
            session.close()
    
    def get_category_stats(self) -> List[Dict[str, Any]]:
        """Get aggregated statistics by category."""
        session = self.get_session()
        try:
            from sqlalchemy import func
            
            stats = session.query(
                ProductTable.category,
                func.count(ProductTable.id).label('count'),
                func.avg(ProductTable.price_min).label('avg_price'),
                func.min(ProductTable.price_min).label('min_price'),
                func.max(ProductTable.price_max).label('max_price')
            ).group_by(ProductTable.category).all()
            
            return [
                {
                    'category': s.category,
                    'count': s.count,
                    'avg_price': round(s.avg_price, 2) if s.avg_price else None,
                    'min_price': s.min_price,
                    'max_price': s.max_price
                }
                for s in stats
            ]
        finally:
            session.close()
    
    def log_scraping_run(self, run_data: Dict[str, Any]) -> int:
        """Log a scraping run for auditing."""
        session = self.get_session()
        try:
            run = ScrapingRunTable(**run_data)
            session.add(run)
            session.commit()
            return run.id
        finally:
            session.close()
    
    def _product_to_dict(self, product: ProductTable) -> Dict[str, Any]:
        """Convert ORM object to dictionary."""
        return {
            'id': product.id,
            'product_name': product.product_name,
            'description': product.description,
            'price_min': product.price_min,
            'price_max': product.price_max,
            'price_raw': product.price_raw,
            'price_unit': product.price_unit,
            'currency': product.currency,
            'supplier_name': product.supplier_name,
            'supplier_verified': product.supplier_verified,
            'supplier_rating': product.supplier_rating,
            'city': product.city,
            'state': product.state,
            'country': product.country,
            'category': product.category,
            'subcategory': product.subcategory,
            'tags': product.tags.split(',') if product.tags else [],
            'source': product.source,
            'source_url': product.source_url,
            'quality_score': product.quality_score,
            'scraped_at': product.scraped_at.isoformat() if product.scraped_at else None
        }
    
    def clear_all_products(self):
        """Clear all products (for testing/reset)."""
        session = self.get_session()
        try:
            session.query(ProductTable).delete()
            session.commit()
            logger.warning("Cleared all products from database")
        finally:
            session.close()


# Singleton instance
db_manager = DatabaseManager()
