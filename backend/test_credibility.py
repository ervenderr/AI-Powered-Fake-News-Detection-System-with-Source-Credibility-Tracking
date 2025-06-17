import unittest
import os
import tempfile
import sqlite3
from pathlib import Path
import sys

# Add the project root to the path so we can import modules correctly
sys.path.append(str(Path(__file__).parent.parent))

from backend.credibility_tracker import CredibilityTracker

class TestCredibilityTracker(unittest.TestCase):
    """Test cases for the CredibilityTracker class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary database file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_credibility.db"
        
        # Create a credibility tracker instance
        self.tracker = CredibilityTracker(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_initialize_db(self):
        """Test database initialization"""
        # Check if database file exists
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check if tables were created
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check sources table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sources'")
        self.assertIsNotNone(cursor.fetchone())
        
        # Check articles table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
        self.assertIsNotNone(cursor.fetchone())
        
        conn.close()
    
    def test_update_source_credibility(self):
        """Test updating source credibility"""
        # Test with a fake article
        domain = "example.com"
        fake_score = self.tracker.update_source_credibility(domain, "fake", 0.8)
        
        # Check if score is appropriate for fake news
        self.assertLess(fake_score, 0.5)
        
        # Test with a real article
        domain = "reliable-news.org"
        real_score = self.tracker.update_source_credibility(domain, "real", 0.9)
        
        # Check if score is appropriate for real news
        self.assertGreater(real_score, 0.5)
        
        # Test updating an existing source
        updated_score = self.tracker.update_source_credibility(domain, "real", 0.7)
        
        # Check if score increased
        self.assertGreaterEqual(updated_score, real_score)
    
    def test_get_source_info(self):
        """Test getting source information"""
        # Test with unknown domain
        domain = "unknown-source.com"
        info = self.tracker.get_source_info(domain)
        
        # Check if default values are returned
        self.assertEqual(info["domain"], domain)
        self.assertEqual(info["total_articles"], 0)
        self.assertEqual(info["credibility_score"], 0.5)
        self.assertEqual(info["status"], "unknown")
        
        # Test with known domain
        domain = "test-source.org"
        self.tracker.update_source_credibility(domain, "real", 0.9)
        info = self.tracker.get_source_info(domain)
        
        # Check if correct values are returned
        self.assertEqual(info["domain"], domain)
        self.assertEqual(info["total_articles"], 1)
        self.assertEqual(info["real_articles"], 1)
        self.assertEqual(info["fake_articles"], 0)
        self.assertGreater(info["credibility_score"], 0.5)
    
    def test_record_article(self):
        """Test recording an article"""
        url = "https://example.com/article1"
        metadata = {
            "domain": "example.com",
            "title": "Test Article",
            "author": "John Doe",
            "publish_date": "2023-01-01T12:00:00"
        }
        
        # Record a fake article
        source_info = self.tracker.record_article(url, metadata, "fake", 0.8)
        
        # Check if source info is returned correctly
        self.assertEqual(source_info["domain"], "example.com")
        self.assertEqual(source_info["total_articles"], 1)
        self.assertEqual(source_info["fake_articles"], 1)
        self.assertEqual(source_info["real_articles"], 0)
        self.assertLess(source_info["credibility_score"], 0.5)
        
        # Check if article was recorded in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT url, domain, title FROM articles WHERE url = ?", (url,))
        article = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(article)
        self.assertEqual(article[0], url)
        self.assertEqual(article[1], "example.com")
        self.assertEqual(article[2], "Test Article")

if __name__ == "__main__":
    unittest.main() 