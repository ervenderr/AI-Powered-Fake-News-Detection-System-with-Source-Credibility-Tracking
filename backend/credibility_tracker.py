import json
import os
import sqlite3
import newspaper
from urllib.parse import urlparse
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend/credibility.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path("data/credibility.db")

class CredibilityTracker:
    def __init__(self, db_path=None):
        """
        Initialize the credibility tracker
        
        Args:
            db_path: Path to the SQLite database file
        """
        if db_path is None:
            db_path = DB_PATH
        
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the SQLite database with necessary tables"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sources table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sources (
            domain TEXT PRIMARY KEY,
            total_articles INTEGER DEFAULT 0,
            fake_articles INTEGER DEFAULT 0,
            real_articles INTEGER DEFAULT 0,
            credibility_score REAL DEFAULT 0.5,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP
        )
        ''')
        
        # Create articles table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            domain TEXT,
            title TEXT,
            author TEXT,
            publish_date TIMESTAMP,
            prediction TEXT,
            confidence REAL,
            processed_date TIMESTAMP,
            FOREIGN KEY (domain) REFERENCES sources (domain)
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def extract_metadata(self, url):
        """
        Extract metadata from a news article URL using newspaper3k
        
        Args:
            url: URL of the news article
            
        Returns:
            dict: Metadata including domain, title, author, and publish date
        """
        try:
            # Parse domain from URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Download and parse article
            article = newspaper.Article(url)
            article.download()
            article.parse()
            
            # Extract metadata
            metadata = {
                "url": url,
                "domain": domain,
                "title": article.title,
                "author": ", ".join(article.authors) if article.authors else None,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None
            }
            
            logger.info(f"Metadata extracted for {url}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {e}")
            
            # Return basic metadata from URL
            return {
                "url": url,
                "domain": parsed_url.netloc if 'parsed_url' in locals() else urlparse(url).netloc,
                "title": None,
                "author": None,
                "publish_date": None
            }
    
    def update_source_credibility(self, domain, prediction, confidence):
        """
        Update the credibility score for a source domain
        
        Args:
            domain: The domain of the news source
            prediction: The prediction label (fake or real)
            confidence: The confidence of the prediction
            
        Returns:
            float: The updated credibility score
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current time
        current_time = datetime.now().isoformat()
        
        # Check if domain exists
        cursor.execute("SELECT * FROM sources WHERE domain = ?", (domain,))
        source = cursor.fetchone()
        
        if source is None:
            # Add new source
            cursor.execute(
                "INSERT INTO sources (domain, total_articles, fake_articles, real_articles, credibility_score, first_seen, last_seen) VALUES (?, 1, ?, ?, ?, ?, ?)",
                (
                    domain,
                    1 if prediction == "fake" else 0,
                    1 if prediction == "real" else 0,
                    0.5 if prediction == "real" else (1.0 - confidence),  # Initial score
                    current_time,
                    current_time
                )
            )
        else:
            # Update existing source
            # Unpack source data
            _, total_articles, fake_articles, real_articles, current_score, first_seen, _ = source
            
            # Update counts
            total_articles += 1
            if prediction == "fake":
                fake_articles += 1
            else:
                real_articles += 1
            
            # Calculate new credibility score
            # Weight by confidence and number of articles
            if total_articles > 0:
                # Base score is the ratio of real to total articles
                base_score = real_articles / total_articles
                
                # Adjust score based on the confidence of this prediction
                # More confident predictions have more weight
                weight = min(0.1 + (0.4 * confidence), 0.5)  # Cap weight at 0.5
                
                # New score is weighted average of current score and prediction
                new_score = (current_score * (1 - weight)) + (
                    weight * (1.0 if prediction == "real" else 0.0)
                )
            else:
                new_score = 0.5  # Default for new sources
            
            # Update source
            cursor.execute(
                "UPDATE sources SET total_articles = ?, fake_articles = ?, real_articles = ?, credibility_score = ?, last_seen = ? WHERE domain = ?",
                (total_articles, fake_articles, real_articles, new_score, current_time, domain)
            )
            
            current_score = new_score
        
        conn.commit()
        
        # Get updated credibility score
        cursor.execute("SELECT credibility_score FROM sources WHERE domain = ?", (domain,))
        updated_score = cursor.fetchone()[0]
        
        conn.close()
        
        logger.info(f"Updated credibility for {domain}: {updated_score:.4f}")
        return updated_score
    
    def record_article(self, url, metadata, prediction, confidence):
        """
        Record an article and update source credibility
        
        Args:
            url: URL of the article
            metadata: Metadata of the article
            prediction: Prediction label (fake or real)
            confidence: Confidence of the prediction
            
        Returns:
            dict: Updated source information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract data from metadata
        domain = metadata.get("domain")
        title = metadata.get("title")
        author = metadata.get("author")
        publish_date = metadata.get("publish_date")
        
        # Get current time
        current_time = datetime.now().isoformat()
        
        # Insert or update article
        cursor.execute(
            "INSERT OR REPLACE INTO articles (url, domain, title, author, publish_date, prediction, confidence, processed_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (url, domain, title, author, publish_date, prediction, confidence, current_time)
        )
        
        conn.commit()
        conn.close()
        
        # Update source credibility
        credibility_score = self.update_source_credibility(domain, prediction, confidence)
        
        # Get source information
        source_info = self.get_source_info(domain)
        
        return source_info
    
    def get_source_info(self, domain):
        """
        Get information about a source
        
        Args:
            domain: Domain of the source
            
        Returns:
            dict: Source information including credibility score and article counts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get source data
        cursor.execute("SELECT * FROM sources WHERE domain = ?", (domain,))
        source = cursor.fetchone()
        
        if source is None:
            # Return default data for unknown source
            source_info = {
                "domain": domain,
                "total_articles": 0,
                "fake_articles": 0,
                "real_articles": 0,
                "credibility_score": 0.5,  # Neutral score for unknown sources
                "first_seen": None,
                "last_seen": None,
                "status": "unknown"
            }
        else:
            # Unpack source data
            _, total_articles, fake_articles, real_articles, credibility_score, first_seen, last_seen = source
            
            # Determine credibility status
            if credibility_score >= 0.7:
                status = "credible"
            elif credibility_score <= 0.3:
                status = "not_credible"
            else:
                status = "uncertain"
            
            source_info = {
                "domain": domain,
                "total_articles": total_articles,
                "fake_articles": fake_articles,
                "real_articles": real_articles,
                "credibility_score": credibility_score,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "status": status
            }
        
        # Get recent articles from this source
        cursor.execute(
            "SELECT url, title, prediction, confidence, processed_date FROM articles WHERE domain = ? ORDER BY processed_date DESC LIMIT 5",
            (domain,)
        )
        articles = cursor.fetchall()
        
        recent_articles = []
        for article in articles:
            url, title, prediction, confidence, processed_date = article
            recent_articles.append({
                "url": url,
                "title": title,
                "prediction": prediction,
                "confidence": confidence,
                "processed_date": processed_date
            })
        
        source_info["recent_articles"] = recent_articles
        
        conn.close()
        
        return source_info
    
    def get_all_sources(self, limit=100, sort_by="credibility_score", ascending=False):
        """
        Get information about all sources
        
        Args:
            limit: Maximum number of sources to return
            sort_by: Field to sort by
            ascending: Whether to sort in ascending order
            
        Returns:
            list: List of source information dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Validate sort field
        valid_sort_fields = ["domain", "total_articles", "fake_articles", "real_articles", "credibility_score", "first_seen", "last_seen"]
        if sort_by not in valid_sort_fields:
            sort_by = "credibility_score"
        
        # Get sources
        order = "ASC" if ascending else "DESC"
        cursor.execute(f"SELECT * FROM sources ORDER BY {sort_by} {order} LIMIT ?", (limit,))
        sources = cursor.fetchall()
        
        result = []
        for source in sources:
            domain, total_articles, fake_articles, real_articles, credibility_score, first_seen, last_seen = source
            
            # Determine credibility status
            if credibility_score >= 0.7:
                status = "credible"
            elif credibility_score <= 0.3:
                status = "not_credible"
            else:
                status = "uncertain"
            
            source_info = {
                "domain": domain,
                "total_articles": total_articles,
                "fake_articles": fake_articles,
                "real_articles": real_articles,
                "credibility_score": credibility_score,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "status": status
            }
            
            result.append(source_info)
        
        conn.close()
        
        return result

# Create a singleton instance
credibility_tracker = CredibilityTracker() 