# Smart Product Review Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.36.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intelligent AI-powered system for detecting fake product reviews and performing multi-aspect sentiment analysis using state-of-the-art NLP techniques.

</div>

---

## 📖 Overview

**Smart Product Review Analyzer** leverages advanced deep learning to help consumers make informed purchasing decisions by detecting potentially fake or artificially generated reviews, analyzing sentiment across multiple product aspects (battery, camera, price, etc.), providing aggregate intelligence from bulk reviews, and calculating trust scores for products.

This project combines **RoBERTa** (Robustly Optimized BERT) for fake detection with **VADER** sentiment analysis for aspect-level insights.

---

##  Features

### Core Capabilities
-  **Fake Review Detection** - Identifies computer-generated or suspicious reviews with 85-92% accuracy
-  **Overall Sentiment Analysis** - Classifies reviews as positive, negative, or neutral with confidence scores
-  **Aspect-Based Sentiment** - Analyzes sentiment for 10 product features (battery, camera, screen, price, etc.)
-  **Bulk Intelligence** - Aggregate analysis across multiple reviews
-  **Trust Scoring** - 0-100 scale authenticity rating
-  **Visual Analytics** - Interactive charts and gauges

### Detected Product Aspects
 Battery |  Camera |  Screen/Display |  Price/Value |  Quality/Build |  Delivery/Shipping |  Design/Aesthetics |  Performance/Speed |  Size/Dimensions | Sound/Audio

---

##  Tech Stack

**Deep Learning Framework:** PyTorch 2.1.0  
**Transformer Model:** RoBERTa-base (125M parameters)  
**NLP Library:** Hugging Face Transformers  
**Sentiment Analysis:** VADER Sentiment  
**Web Framework:** Streamlit  
**Visualization:** Plotly, Plotly Express  
**Data Processing:** Pandas, NumPy  
**ML Utilities:** scikit-learn

---

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/smart-review-analyzer.git
cd smart-review-analyzer
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download NLTK data**
```bash
python -m textblob.download_corpora
```

**5. Download dataset**
Download fake reviews dataset from [Kaggle](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset) and place `fake_reviews_dataset.csv` in the project root.

---

##  Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 85-92% |
| Precision | 0.84-0.90 |
| Recall | 0.82-0.88 |
| F1 Score | 0.83-0.89 |

---


## 🤝 Contributing

Contributions are welcome! Fork the repository, create a feature branch, commit changes, and open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

##  Acknowledgments

- Dataset: [Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)
- RoBERTa Model: [Hugging Face](https://huggingface.co/roberta-base)
- VADER Sentiment: [vaderSentiment](https://github.com/cjhutto/vaderSentiment)

---

<div align="center">

If you found this helpful, please ⭐ the repo!
