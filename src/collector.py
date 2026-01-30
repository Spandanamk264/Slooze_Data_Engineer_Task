"""
Slooze Data Engineering Challenge - Advanced Data Collector
============================================================

Production-grade, async-capable web scraper with:
- Concurrent request handling
- Intelligent rate limiting
- User agent rotation
- Retry logic with exponential backoff
- Data validation at ingestion
- Multiple source support
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
import random
import re
import logging
import json
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from fake_useragent import UserAgent
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import scraper_config, DATA_DIR
from models import ProductListing, DataSource
from database import db_manager

# Setup
console = Console()
logger = logging.getLogger(__name__)

# User agent rotation
try:
    ua = UserAgent()
except:
    ua = None


class RateLimiter:
    """Token bucket rate limiter for respectful scraping."""
    
    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute / 60.0  # Convert to per second
        self.tokens = requests_per_minute
        self.max_tokens = requests_per_minute
        self.last_update = datetime.now()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = datetime.now()
            elapsed = (now - self.last_update).total_seconds()
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class BaseParser:
    """Base class for marketplace-specific parsers."""
    
    def parse(self, html: str, category: str) -> List[Dict[str, Any]]:
        """Parse HTML and return list of product dictionaries."""
        raise NotImplementedError


class IndiaMARTParser(BaseParser):
    """Parser for IndiaMART marketplace."""
    
    def parse(self, html: str, category: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html, 'lxml')
        products = []
        
        # Multiple selector strategies for resilience
        selectors = [
            'div.c-p',
            'div.prd-card',
            'li.prd-li',
            'div[data-ggl]',
        ]
        
        listings = []
        for selector in selectors:
            listings = soup.select(selector)
            if listings:
                logger.debug(f"Found {len(listings)} listings with selector: {selector}")
                break
        
        for item in listings:
            try:
                product = self._parse_listing(item, category)
                if product:
                    products.append(product)
            except Exception as e:
                logger.debug(f"Failed to parse listing: {e}")
                continue
                
        return products
    
    def _parse_listing(self, item, category: str) -> Optional[Dict[str, Any]]:
        """Parse individual listing."""
        # Name extraction
        name_elem = (
            item.select_one('a.pn') or 
            item.select_one('.prd-name') or
            item.select_one('h2 a') or
            item.select_one('a[title]')
        )
        if not name_elem:
            return None
        
        name = name_elem.get_text(strip=True) or name_elem.get('title', '')
        
        # Price extraction
        price_elem = item.select_one('.prc') or item.select_one('.price')
        price_raw = price_elem.get_text(strip=True) if price_elem else None
        price_min, price_max = self._parse_price_range(price_raw)
        
        # Supplier
        supplier_elem = item.select_one('.cname') or item.select_one('.company-name')
        supplier = supplier_elem.get_text(strip=True) if supplier_elem else None
        
        # Location
        loc_elem = item.select_one('.cloc') or item.select_one('.location')
        city = loc_elem.get_text(strip=True) if loc_elem else None
        
        # URL
        link = name_elem.get('href', '') if name_elem.name == 'a' else None
        
        return {
            'product_name': name,
            'price_min': price_min,
            'price_max': price_max,
            'price_raw': price_raw,
            'supplier_name': supplier,
            'city': city,
            'category': category.replace('-', ' ').title(),
            'source': DataSource.INDIAMART.value,
            'source_url': link,
            'scraped_at': datetime.now()
        }
    
    def _parse_price_range(self, price_str: Optional[str]) -> tuple:
        """Extract min/max price from price string."""
        if not price_str:
            return None, None
        
        # Find all numbers in the string
        numbers = re.findall(r'[\d,]+\.?\d*', price_str.replace(',', ''))
        numbers = [float(n) for n in numbers if n]
        
        if len(numbers) >= 2:
            return min(numbers), max(numbers)
        elif len(numbers) == 1:
            return numbers[0], numbers[0]
        return None, None


class MockDataGenerator:
    """
    Generates realistic mock data for demonstration.
    
    Used when actual scraping is blocked or for testing.
    Produces statistically realistic distributions.
    """
    
    PRODUCTS = {
        'industrial-pumps': [
            'Centrifugal Pump', 'Submersible Pump', 'Hydraulic Pump',
            'Vacuum Pump', 'Gear Pump', 'Diaphragm Pump', 'Screw Pump',
            'Piston Pump', 'Vane Pump', 'Rotary Pump'
        ],
        'electric-motors': [
            'AC Induction Motor', 'DC Motor', 'Servo Motor', 'Stepper Motor',
            'Brushless DC Motor', 'Synchronous Motor', 'Linear Motor',
            'Universal Motor', 'Reluctance Motor', 'Hysteresis Motor'
        ],
        'cnc-machines': [
            'CNC Lathe Machine', 'CNC Milling Machine', 'CNC Router',
            'CNC Plasma Cutter', 'CNC Laser Cutter', 'CNC Grinding Machine',
            'Wire EDM Machine', 'CNC Drilling Machine', '5-Axis CNC',
            'CNC Turning Center'
        ],
        'textile-machinery': [
            'Spinning Machine', 'Weaving Loom', 'Knitting Machine',
            'Dyeing Machine', 'Printing Machine', 'Finishing Machine',
            'Embroidery Machine', 'Cutting Machine', 'Sewing Machine',
            'Inspection Machine'
        ],
        'packaging-machines': [
            'Filling Machine', 'Sealing Machine', 'Wrapping Machine',
            'Labeling Machine', 'Carton Erector', 'Palletizer',
            'Shrink Wrapper', 'Vacuum Packer', 'Blister Packer',
            'Case Packer'
        ]
    }
    
    SUPPLIERS = [
        'Apex Engineering Works', 'Global Industrial Solutions', 
        'TechPro Machineries', 'Standard Equipment Co.', 'Premier Industries',
        'Excel Manufacturing', 'Precision Tools Ltd.', 'Quality Engineering',
        'Supreme Machines', 'Advanced Systems Inc.', 'Delta Industrial',
        'Sigma Engineering', 'Omega Machineries', 'Prime Equipment',
        'Alpha Industries', 'Beta Manufacturing', 'Gamma Tools',
        'Zeta Engineering', 'Theta Industries', 'Iota Systems'
    ]
    
    CITIES = [
        'Mumbai', 'Delhi', 'Ahmedabad', 'Chennai', 'Pune', 'Coimbatore',
        'Bangalore', 'Hyderabad', 'Kolkata', 'Rajkot', 'Ludhiana', 'Surat',
        'Jaipur', 'Vadodara', 'Nagpur', 'Indore', 'Faridabad', 'Ghaziabad'
    ]
    
    STATES = {
        'Mumbai': 'Maharashtra', 'Pune': 'Maharashtra', 'Nagpur': 'Maharashtra',
        'Delhi': 'Delhi', 'Faridabad': 'Haryana', 'Ghaziabad': 'Uttar Pradesh',
        'Ahmedabad': 'Gujarat', 'Rajkot': 'Gujarat', 'Surat': 'Gujarat', 'Vadodara': 'Gujarat',
        'Chennai': 'Tamil Nadu', 'Coimbatore': 'Tamil Nadu',
        'Bangalore': 'Karnataka', 'Hyderabad': 'Telangana',
        'Kolkata': 'West Bengal', 'Ludhiana': 'Punjab',
        'Jaipur': 'Rajasthan', 'Indore': 'Madhya Pradesh'
    }
    
    # Price ranges by category (realistic INR values)
    PRICE_RANGES = {
        'industrial-pumps': (5000, 500000),
        'electric-motors': (2000, 300000),
        'cnc-machines': (500000, 5000000),
        'textile-machinery': (100000, 2000000),
        'packaging-machines': (50000, 1500000)
    }
    
    def generate(self, categories: List[str], count_per_category: int = 50) -> List[Dict[str, Any]]:
        """Generate mock product data."""
        products = []
        
        for category in categories:
            category_products = self.PRODUCTS.get(category, ['Generic Product'])
            price_range = self.PRICE_RANGES.get(category, (10000, 500000))
            
            for _ in range(count_per_category):
                city = random.choice(self.CITIES)
                base_product = random.choice(category_products)
                
                # Add realistic variation to product names
                model_suffix = f"Model {random.choice(['A', 'B', 'C', 'X', 'Z'])}{random.randint(100, 999)}"
                capacity_suffix = random.choice(['', ' - Heavy Duty', ' - Standard', ' - Industrial Grade', ' Pro'])
                
                # Generate price with realistic distribution
                base_price = random.uniform(*price_range)
                # Apply log-normal distribution for more realistic spread
                price_factor = random.lognormvariate(0, 0.3)
                price_min = round(base_price * price_factor, -2)  # Round to nearest 100
                price_variation = random.uniform(1.0, 1.3)
                price_max = round(price_min * price_variation, -2)
                
                products.append({
                    'product_name': f"{base_product} {model_suffix}{capacity_suffix}",
                    'description': f"High-quality {base_product.lower()} suitable for industrial applications. "
                                   f"Features advanced engineering and durable construction.",
                    'price_min': price_min,
                    'price_max': price_max,
                    'price_raw': f"₹ {int(price_min):,} - ₹ {int(price_max):,}",
                    'price_unit': random.choice(['per_piece', 'per_unit', 'per_set']),
                    'currency': 'INR',
                    'supplier_name': random.choice(self.SUPPLIERS),
                    'supplier_verified': random.random() > 0.3,
                    'supplier_rating': round(random.uniform(3.5, 5.0), 1),
                    'city': city,
                    'state': self.STATES.get(city, 'Maharashtra'),
                    'country': 'India',
                    'category': category.replace('-', ' ').title(),
                    'subcategory': base_product,
                    'tags': [category, base_product.lower(), 'industrial', 'machinery'],
                    'source': DataSource.MOCK.value,
                    'source_url': 'mock_generated',
                    'product_id': f"MOCK-{random.randint(10000, 99999)}",
                    'scraped_at': datetime.now(),
                    'quality_score': 1.0,
                    'is_complete': True
                })
        
        random.shuffle(products)
        return products


class AdvancedDataCollector:
    """
    Advanced data collection system with async support.
    
    Features:
    - Async HTTP requests for performance
    - Rate limiting for respectful scraping
    - Automatic fallback to mock data
    - Data validation with Pydantic
    - Database persistence
    - Detailed logging and progress tracking
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter(scraper_config.RATE_LIMIT)
        self.parsers = {
            'indiamart': IndiaMARTParser()
        }
        self.mock_generator = MockDataGenerator()
        self.collected_data: List[Dict[str, Any]] = []
        
    def get_headers(self) -> Dict[str, str]:
        """Get request headers with optional UA rotation."""
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        if scraper_config.ROTATE_USER_AGENTS and ua:
            headers['User-Agent'] = ua.random
        else:
            headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            
        return headers
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.RequestException, aiohttp.ClientError))
    )
    async def fetch_url_async(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch URL with retry logic."""
        await self.rate_limiter.acquire()
        
        try:
            async with session.get(url, headers=self.get_headers(), timeout=15) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 403:
                    logger.warning(f"Blocked (403) on {url}")
                    return None
                else:
                    logger.warning(f"HTTP {response.status} on {url}")
                    return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout on {url}")
            return None
    
    async def scrape_category_async(
        self, 
        session: aiohttp.ClientSession,
        source: str,
        category: str
    ) -> List[Dict[str, Any]]:
        """Scrape a single category."""
        base_url = scraper_config.SOURCES.get(source)
        if not base_url:
            return []
        
        url = f"{base_url}/impcat/{category}.html"
        logger.info(f"Scraping: {url}")
        
        html = await self.fetch_url_async(session, url)
        if not html:
            return []
        
        parser = self.parsers.get(source)
        if parser:
            return parser.parse(html, category)
        
        return []
    
    async def run_async(self, use_mock: bool = False) -> List[Dict[str, Any]]:
        """Run the async collection pipeline."""
        console.print("[bold blue]Starting Advanced Data Collection Pipeline[/bold blue]")
        
        if use_mock:
            return self._generate_mock_data()
        
        all_products = []
        
        async with aiohttp.ClientSession() as session:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(
                    "[cyan]Scraping categories...", 
                    total=len(scraper_config.CATEGORIES)
                )
                
                for category in scraper_config.CATEGORIES:
                    products = await self.scrape_category_async(session, 'indiamart', category)
                    all_products.extend(products)
                    progress.advance(task)
                    
                    # Small delay between categories
                    await asyncio.sleep(random.uniform(1, 2))
        
        if not all_products:
            console.print("[yellow]>> No data scraped. Falling back to mock data generation.[/yellow]")
            return self._generate_mock_data()
        
        return all_products
    
    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock data as fallback."""
        console.print("[cyan]Generating realistic mock data for demonstration...[/cyan]")
        return self.mock_generator.generate(
            scraper_config.CATEGORIES,
            count_per_category=60
        )
    
    def validate_and_store(self, products: List[Dict[str, Any]]) -> int:
        """Validate products and store in database."""
        validated = []
        rejected = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Validating data...", total=len(products))
            
            for product_data in products:
                try:
                    # Validate with Pydantic
                    validated_product = ProductListing(**product_data)
                    validated.append(validated_product.model_dump())
                except Exception as e:
                    logger.debug(f"Validation failed: {e}")
                    rejected += 1
                    
                progress.advance(task)
        
        console.print(f"[green]+ Validated: {len(validated)}[/green] | [red]x Rejected: {rejected}[/red]")
        
        # Store in database
        if validated:
            db_manager.insert_products(validated)
            
            # Also save to CSV for easy viewing
            import pandas as pd
            df = pd.DataFrame(validated)
            csv_path = DATA_DIR / "collected_data.csv"
            df.to_csv(csv_path, index=False)
            console.print(f"[dim]Data also saved to {csv_path}[/dim]")
        
        return len(validated)
    
    def run(self, use_mock: bool = False) -> int:
        """
        Main entry point - runs the collection pipeline.
        
        Returns:
            Number of records collected and stored
        """
        start_time = datetime.now()
        
        # Run async collection
        products = asyncio.run(self.run_async(use_mock))
        
        # Validate and store
        count = self.validate_and_store(products)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        console.print(f"\n[bold green]Collection complete![/bold green]")
        console.print(f"   Records: {count} | Time: {elapsed:.1f}s")
        
        return count


def main():
    """CLI entry point."""
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/collector.log'),
            logging.StreamHandler()
        ]
    )
    
    collector = AdvancedDataCollector()
    
    # Check for --mock flag
    use_mock = '--mock' in sys.argv
    
    collector.run(use_mock=use_mock)


if __name__ == "__main__":
    main()
