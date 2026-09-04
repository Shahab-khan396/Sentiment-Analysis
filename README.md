# ✨ AI Sentiment Analyzer & Feature Extractor

An interactive, cloud-ready Natural Language Processing (NLP) web application for **Sentiment Analysis** and **Keyword Extraction**, powered by **Streamlit**, **VADER Sentiment Analysis**, and **Scikit-learn TF-IDF**.

---

## 🚀 1-Click Deployment to Streamlit Community Cloud (Free)

This repository is pre-configured for instant zero-configuration deployment on **Streamlit Community Cloud**:

### 1. Push to GitHub
```bash
git add .
git commit -m "Deploy AI Sentiment Analyzer on Streamlit"
git push origin main
```

### 2. Launch on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository: `Sentiment-Analysis`.
4. Branch: `main`
5. Main file path: `streamlit_app.py`
6. Click **"Deploy!"** 🎈

Your application will be live on a public URL in less than 2 minutes.

---

## 🌟 Key Features

- **Accurate Polarity & Valence Scoring**: Uses VADER sentiment analysis, calibrated to social context, emoticons, and valence shifters.
- **Negation & Modifier Preservation**: Properly accounts for negation words (e.g., *"not good"*, *"never fails"*) without stripping sentiment-bearing terms.
- **Interactive Visualizations**:
  - **Compound Polarity Gauge**: High-impact Plotly gauge ranging from $-1.00$ to $+1.00$.
  - **Sentiment Ratio Donut Chart**: Breakdown of Positive, Neutral, and Negative proportions.
- **Scikit-learn TF-IDF Keyword Extraction**: Automatically discovers and ranks the most influential n-grams and terms.
- **Batch CSV / Dataset Processing**:
  - Upload `.csv` or `.txt` files containing customer reviews or survey comments.
  - Automatically classifies all rows and computes overall dataset distribution.
  - Export classified data with sentiment tags via **"Download Analyzed CSV"**.
  - Built-in sample dataset generator for instant testing.
- **One-Click Test Presets**: Easily test with pre-built positive, negative, neutral, and subtle negation samples.
- **Sleek Dark Theme**: Configured with `.streamlit/config.toml` and Google Fonts (`Outfit`).
- **Cloud Caching**: Optimized with `@st.cache_resource` for zero-latency instant startup.

---

## 💻 Local Setup & Execution

### Prerequisites
Python 3.10+ installed.

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Shahab-khan396/Sentiment-Analysis.git
cd Sentiment-Analysis
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run streamlit_app.py
```

The app will open automatically in your browser at **`http://localhost:8501`**.

---

## 📁 Repository Structure

```
Sentiment-Analysis/
├── .streamlit/
│   └── config.toml        # Custom dark theme and server settings
├── .gitignore             # Standard Python / venv ignore rules
├── LICENSE                # MIT License
├── README.md              # Project documentation & deployment guide
├── requirements.txt       # Dependencies for Streamlit Cloud deployment
└── streamlit_app.py       # Main Streamlit application entrypoint
```

---

## 📜 Dependencies

- `streamlit` - Web application framework
- `vaderSentiment` - Rule-based sentiment analysis
- `nltk` - Natural language toolkit (corpus stopwords)
- `scikit-learn` - TF-IDF feature extraction
- `pandas` - Tabular data processing for batch CSV analysis
- `plotly` - Interactive gauges and donut charts

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).