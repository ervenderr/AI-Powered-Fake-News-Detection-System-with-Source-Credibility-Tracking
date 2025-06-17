# AI-Powered Fake News Detection System

This project implements an AI-powered system for detecting fake news and tracking source credibility. It includes a machine learning model for classifying news articles and a browser extension for real-time detection.

## Project Structure

```
.
├── backend/           # FastAPI server code
├── frontend/          # Chrome extension code
├── model/             # ML model training and inference code
├── data/              # Datasets and preprocessing scripts
│   ├── raw/           # Raw downloaded datasets
│   └── processed/     # Processed datasets for training
└── venv/              # Python virtual environment
```

## Setup Instructions

1. **Clone the repository**

   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set up the Python environment**

   ```
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download and preprocess datasets**

   ```
   python data/download_datasets.py
   python data/preprocess_datasets.py
   ```

4. **Run the API server**

   ```
   uvicorn backend.main:app --reload
   ```

5. **Test the API**
   ```
   python backend/test_api.py
   ```

## Datasets

This project uses the following datasets:

- **LIAR Dataset**: A benchmark dataset for fake news detection with 12.8K human-labeled short statements.
- **FakeNewsNet**: A comprehensive collection of fake news articles with social context information.

## Features

- Fake news detection using transformer-based models
- Source credibility tracking
- Browser extension for real-time detection
- Feedback mechanism for continuous improvement

## License

[MIT License](LICENSE)
