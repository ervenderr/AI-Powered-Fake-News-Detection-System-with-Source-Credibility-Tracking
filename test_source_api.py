import requests
import json
import time

def test_prediction_with_url():
    """Test the prediction endpoint with a URL"""
    url = "http://127.0.0.1:8000/predict"
    
    test_cases = [
        {
            "text": "Scientists have discovered that drinking water is actually bad for your health.",
            "url": "https://fake-news-site.com/water-bad-health"
        },
        {
            "text": "A new study shows that regular exercise can reduce the risk of heart disease.",
            "url": "https://real-news-site.org/exercise-heart-health"
        }
    ]
    
    print("Testing prediction endpoint with URLs...")
    print("-" * 50)
    
    for case in test_cases:
        try:
            response = requests.post(url, json=case)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Text: {case['text']}")
                print(f"URL: {case['url']}")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}")
                
                if "source_info" in result:
                    source_info = result["source_info"]
                    print(f"Source: {source_info['domain']}")
                    print(f"Credibility Score: {source_info['credibility_score']:.4f}")
                    print(f"Status: {source_info['status']}")
                    print(f"Articles: {source_info['total_articles']} (Real: {source_info['real_articles']}, Fake: {source_info['fake_articles']})")
                
                print("-" * 50)
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Request failed: {e}")
    
    # Allow time for the database to update
    time.sleep(1)

def test_source_score():
    """Test the source score endpoint"""
    base_url = "http://127.0.0.1:8000/source_score"
    
    domains = [
        "fake-news-site.com",
        "real-news-site.org",
        "unknown-domain.net"
    ]
    
    print("Testing source score endpoint...")
    print("-" * 50)
    
    for domain in domains:
        try:
            response = requests.get(f"{base_url}?domain={domain}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Domain: {result['domain']}")
                print(f"Credibility Score: {result['credibility_score']:.4f}")
                print(f"Status: {result['status']}")
                print(f"Articles: {result['total_articles']} (Real: {result['real_articles']}, Fake: {result['fake_articles']})")
                
                if "recent_articles" in result and result["recent_articles"]:
                    print("Recent Articles:")
                    for article in result["recent_articles"]:
                        print(f"  - {article['title'] or article['url']} ({article['prediction']})")
                
                print("-" * 50)
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Request failed: {e}")

def test_all_sources():
    """Test the all sources endpoint"""
    url = "http://127.0.0.1:8000/sources"
    
    print("Testing all sources endpoint...")
    print("-" * 50)
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            sources = response.json()
            print(f"Found {len(sources)} sources:")
            
            for source in sources:
                print(f"Domain: {source['domain']}")
                print(f"Credibility Score: {source['credibility_score']:.4f}")
                print(f"Status: {source['status']}")
                print(f"Articles: {source['total_articles']} (Real: {source['real_articles']}, Fake: {source['fake_articles']})")
                print("-" * 30)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    print("Testing source credibility API...")
    print("=" * 50)
    
    test_prediction_with_url()
    test_source_score()
    test_all_sources()
    
    print("API testing completed.") 