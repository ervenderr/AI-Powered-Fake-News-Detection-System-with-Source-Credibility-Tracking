import requests
import json

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/ping")
        if response.status_code == 200:
            print("✅ Health check endpoint is working!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        print("Make sure the API server is running with: uvicorn backend.main:app --reload")

if __name__ == "__main__":
    print("Testing API endpoints...")
    test_health_check() 