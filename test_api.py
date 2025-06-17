import requests
import json

def test_prediction():
    """Test the fake news prediction endpoint"""
    url = "http://127.0.0.1:8000/predict"
    
    test_texts = [
        "Scientists have discovered that drinking water is actually bad for your health.",
        "A new study shows that regular exercise can reduce the risk of heart disease.",
        "The government has secretly been putting microchips in vaccines to track citizens.",
        "The stock market closed higher yesterday due to positive economic data."
    ]
    
    print("Testing fake news detection API...")
    print("-" * 50)
    
    for text in test_texts:
        payload = {"text": text}
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Text: {text}")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Fake probability: {result['fake_probability']:.4f}")
                print(f"Real probability: {result['real_probability']:.4f}")
                print("-" * 50)
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Request failed: {e}")
    
    print("API testing completed.")

if __name__ == "__main__":
    test_prediction() 