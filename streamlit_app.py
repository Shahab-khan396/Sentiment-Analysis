"""AI Sentiment Analysis & Keyword Extraction Application
Built with Streamlit, VADER Sentiment, and Scikit-learn TF-IDF.
Optimized for one-click deployment on Streamlit Community Cloud and Docker/HuggingFace.
"""

import re
import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(16, 185, 129, 0.08) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    .badge-keyword {
        display: inline-block;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #e0e7ff;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        margin: 0.25rem;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Cached NLP Resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_nlp_resources():
    """Download and cache stopwords and VADER analyzer for cloud cold starts."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words('english'))
    negation_words = {
        'not', 'no', 'nor', 'never', 'none', 'neither', 'without',
        'hardly', 'barely', 'scarcely', "isn't", "aren't", "wasn't",
        "weren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't",
        "don't", "doesn't", "didn't", "can't", "couldn't", "shouldn't",
        "mustn't", "ain't"
    }
    filtered_stopwords = stop_words - negation_words
    analyzer = SentimentIntensityAnalyzer()
    return analyzer, filtered_stopwords

analyzer, filtered_stopwords = load_nlp_resources()


# -----------------------------------------------------------------------------
# 3. Core NLP Helper Functions
# -----------------------------------------------------------------------------
def analyze_single_text(text: str, pos_threshold=0.05, neg_threshold=-0.05):
    """Analyze a single text string with VADER and TF-IDF feature extraction."""
    raw_text = text.strip()
    if not raw_text:
        return None

    scores = analyzer.polarity_scores(text=raw_text)
    compound = scores['compound']

    if compound >= pos_threshold:
        label = "Positive"
        css_class = "positive"
        icon = "😊"
        color = "#10b981"
    elif compound <= neg_threshold:
        label = "Negative"
        css_class = "negative"
        icon = "😞"
        color = "#f43f5e"
    else:
        label = "Neutral"
        css_class = "neutral"
        icon = "😐"
        color = "#f59e0b"

    positivity_pct = round((1 + compound) * 50, 1)
    keywords = extract_keywords([raw_text], top_n=6)

    return {
        "raw_text": raw_text,
        "label": label,
        "css_class": css_class,
        "icon": icon,
        "color": color,
        "compound": round(compound, 4),
        "positivity_pct": positivity_pct,
        "pos": round(scores['pos'] * 100, 1),
        "neu": round(scores['neu'] * 100, 1),
        "neg": round(scores['neg'] * 100, 1),
        "keywords": keywords,
    }


def extract_keywords(corpus, top_n=6):
    """Extract top salient n-grams using Scikit-learn TfidfVectorizer."""
    try:
        combined = " ".join(corpus)
        clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', combined).strip()
        words = [w for w in clean.split() if len(w) > 2]
        if not words:
            return []
        if len(words) <= 2:
            return [{"term": w.lower(), "score": 1.0} for w in words]

        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=top_n
        )
        tfidf_matrix = vectorizer.fit_transform([clean])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        ranked = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
        return [{"term": term, "score": round(score, 3)} for term, score in ranked if score > 0]
    except Exception:
        return []


def create_gauge_chart(compound_score: float):
    """Create a Plotly gauge for the VADER Compound Score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=compound_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "VADER Compound Polarity", 'font': {'size': 18, 'color': '#94a3b8'}},
        number={'font': {'size': 32, 'color': '#ffffff'}, 'valueformat': "+.3f"},
        gauge={
            'axis': {'range': [-1.0, 1.0], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': "#818cf8", 'thickness': 0.25},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [-1.0, -0.05], 'color': 'rgba(244, 63, 94, 0.25)'},
                {'range': [-0.05, 0.05], 'color': 'rgba(245, 158, 11, 0.25)'},
                {'range': [0.05, 1.0], 'color': 'rgba(16, 185, 129, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': compound_score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f8fafc", 'family': "Outfit"},
        margin=dict(l=20, r=20, t=40, b=20),
        height=240,
    )
    return fig


def create_donut_chart(pos: float, neu: float, neg: float):
    """Create an interactive donut chart showing sentiment breakdown."""
    labels = ['Positive', 'Neutral', 'Negative']
    values = [pos, neu, neg]
    colors = ['#10b981', '#60a5fa', '#f43f5e']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.55,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hoverinfo='label+value',
    )])
    fig.update_layout(
        title={'text': "Sentiment Proportion", 'font': {'size': 18, 'color': '#94a3b8'}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f8fafc", 'family': "Outfit"},
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=240,
    )
    return fig


# -----------------------------------------------------------------------------
# 4. Main App Layout & Logic
# -----------------------------------------------------------------------------
def render_app():
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")
        pos_cutoff = st.slider("Positive Threshold (Compound ≥)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
        neg_cutoff = st.slider("Negative Threshold (Compound ≤)", min_value=-0.5, max_value=0.0, value=-0.05, step=0.01)

        st.markdown("---")
        st.markdown("### 🧠 Model Architecture")
        st.info(
            """
            - **VADER**: Lexicon and rule-based sentiment engine attuned to negations and valence shifters.
            - **Scikit-learn TF-IDF**: Extracts and ranks key n-grams based on Term Frequency & Inverse Document Frequency.
            """
        )

        st.markdown("---")
        st.markdown("### 🚀 Easy 1-Click Deployment")
        st.markdown(
            """
            1. Push this code to **GitHub**.
            2. Visit **[share.streamlit.io](https://share.streamlit.io)**.
            3. Select repository, branch, and entrypoint `streamlit_app.py`.
            4. Click **Deploy**! ✨
            """
        )

    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">✨ AI Sentiment Analyzer & Feature Extractor</div>
        <div class="hero-subtitle">
            Evaluate polarity and discover thematic keywords using <b>VADER Sentiment</b> and <b>Scikit-learn TF-IDF</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(["✍️ Single Text Analysis", "📁 Batch CSV / Dataset Analysis"])

    # TAB 1: Single Text Analysis
    with tab1:
        st.markdown("#### Choose an Example or Enter Text")

        examples = {
            "🌟 Positive Review": "This product completely blew me away! Incredible build quality, blisteringly fast shipping, and friendly support. 10/10!",
            "⚠️ Negative Feedback": "Worst customer service I have ever dealt with. The item arrived broken in pieces and they refused to honor the refund guarantee.",
            "📄 Neutral Fact": "The parcel was delivered on Tuesday afternoon and contained three standard notebook pads and a ballpoint pen.",
            "🔍 Negation Test": "I was genuinely worried this wouldn't work, but it is not bad at all. In fact, it is surprisingly pleasant!"
        }

        cols = st.columns(len(examples))
        for i, (title, text_val) in enumerate(examples.items()):
            if cols[i].button(title, use_container_width=True):
                st.session_state["user_input_text"] = text_val

        if "user_input_text" not in st.session_state:
            st.session_state["user_input_text"] = examples["🌟 Positive Review"]

        input_text = st.text_area(
            label="Input Text:",
            value=st.session_state["user_input_text"],
            height=130,
            placeholder="Type, paste, or select an example above...",
            key="main_textarea"
        )

        col_btn, _ = st.columns([2, 10])
        with col_btn:
            st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

        if input_text:
            res = analyze_single_text(input_text, pos_threshold=pos_cutoff, neg_threshold=neg_cutoff)

            if res:
                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        label="Classification",
                        value=f"{res['icon']} {res['label']}",
                        delta=f"{res['compound']:+.3f} VADER",
                    )
                with c2:
                    st.metric(
                        label="Positivity Index",
                        value=f"{res['positivity_pct']}%",
                    )
                with c3:
                    st.metric(
                        label="Positive Ratio",
                        value=f"{res['pos']}%",
                    )
                with c4:
                    st.metric(
                        label="Negative Ratio",
                        value=f"{res['neg']}%",
                    )

                col_gauge, col_donut = st.columns(2)
                with col_gauge:
                    st.plotly_chart(create_gauge_chart(res['compound']), use_container_width=True)
                with col_donut:
                    st.plotly_chart(create_donut_chart(res['pos'], res['neu'], res['neg']), use_container_width=True)

                if res['keywords']:
                    st.markdown("##### 🔑 Scikit-learn TF-IDF Key Extracted Terms")
                    kw_html = "".join(
                        [f'<span class="badge-keyword"><b>{k["term"]}</b> <small>({k["score"]})</small></span>' for k in res['keywords']]
                    )
                    st.markdown(f"<div>{kw_html}</div>", unsafe_allow_html=True)

                with st.expander("🔍 View Raw Score Breakdown"):
                    st.json({
                        "compound_score": res['compound'],
                        "positivity_percent": res['positivity_pct'],
                        "positive_ratio": res['pos'] / 100.0,
                        "neutral_ratio": res['neu'] / 100.0,
                        "negative_ratio": res['neg'] / 100.0,
                        "text_length": len(res['raw_text']),
                        "keywords": res['keywords']
                    })
        else:
            st.warning("Please type or paste some text to analyze.")

    # TAB 2: Batch CSV / Dataset Analysis
    with tab2:
        st.markdown("#### Bulk Analyze Customer Reviews, Tweets, or Survey Comments")
        st.write("Upload a CSV file or text dataset to classify multiple rows simultaneously.")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "txt"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded successfully! Loaded {len(df)} rows.")

                text_cols = [col for col in df.columns if df[col].dtype == 'object']
                if not text_cols:
                    text_cols = list(df.columns)

                selected_col = st.selectbox("Select the column containing text/reviews to analyze:", text_cols)

                if st.button("⚡ Run Batch Analysis", type="primary"):
                    with st.spinner("Analyzing sentiments across dataset..."):
                        compounds = []
                        labels = []
                        pos_list = []
                        neu_list = []
                        neg_list = []

                        for val in df[selected_col].astype(str):
                            sc = analyzer.polarity_scores(val)
                            cmp = sc['compound']
                            compounds.append(cmp)
                            pos_list.append(sc['pos'])
                            neu_list.append(sc['neu'])
                            neg_list.append(sc['neg'])

                            if cmp >= pos_cutoff:
                                labels.append("Positive")
                            elif cmp <= neg_cutoff:
                                labels.append("Negative")
                            else:
                                labels.append("Neutral")

                        df['sentiment_label'] = labels
                        df['compound_score'] = compounds
                        df['positive_ratio'] = pos_list
                        df['neutral_ratio'] = neu_list
                        df['negative_ratio'] = neg_list

                        st.markdown("---")
                        st.markdown("### 📊 Dataset Sentiment Insights")

                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        total_count = len(df)
                        pos_count = (df['sentiment_label'] == 'Positive').sum()
                        neu_count = (df['sentiment_label'] == 'Neutral').sum()
                        neg_count = (df['sentiment_label'] == 'Negative').sum()
                        avg_compound = df['compound_score'].mean()

                        col_m1.metric("Total Analyzed", total_count)
                        col_m2.metric("Positive %", f"{round(pos_count/total_count*100, 1)}%")
                        col_m3.metric("Negative %", f"{round(neg_count/total_count*100, 1)}%")
                        col_m4.metric("Avg Compound", f"{avg_compound:+.3f}")

                        fig_dist = px.pie(
                            names=['Positive', 'Neutral', 'Negative'],
                            values=[pos_count, neu_count, neg_count],
                            color=['Positive', 'Neutral', 'Negative'],
                            color_discrete_map={'Positive': '#10b981', 'Neutral': '#60a5fa', 'Negative': '#f43f5e'},
                            hole=0.45,
                            title="Overall Sentiment Distribution"
                        )
                        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'family': 'Outfit'})
                        st.plotly_chart(fig_dist, use_container_width=True)

                        corpus_sample = df[selected_col].astype(str).tolist()[:500]
                        top_kw = extract_keywords(corpus_sample, top_n=10)
                        if top_kw:
                            st.markdown("##### 🏆 Top Extracted Keywords Across Dataset (Scikit-learn TF-IDF)")
                            kw_html = "".join(
                                [f'<span class="badge-keyword"><b>{k["term"]}</b> <small>({k["score"]})</small></span>' for k in top_kw]
                            )
                            st.markdown(f"<div>{kw_html}</div>", unsafe_allow_html=True)

                        st.markdown("##### 📄 Classified Results")
                        st.dataframe(df, use_container_width=True)

                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Analyzed CSV",
                            data=csv_buffer.getvalue(),
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.info("💡 Don't have a CSV handy? Click below to load a sample review dataset:")
            if st.button("Load Sample Reviews Dataset"):
                sample_data = {
                    "Review_ID": [101, 102, 103, 104, 105],
                    "Customer_Feedback": [
                        "Absolutely thrilled with the high quality and prompt delivery!",
                        "Terrible experience, device stopped working on the second day.",
                        "Package arrived on time, contents were as described in the catalogue.",
                        "Customer service was not helpful at all, very frustrating.",
                        "It is not bad, does the job well enough for the price point."
                    ]
                }
                sample_df = pd.DataFrame(sample_data)
                st.dataframe(sample_df, use_container_width=True)
                csv_sample = io.StringIO()
                sample_df.to_csv(csv_sample, index=False)
                st.download_button(
                    label="Download Sample CSV Template",
                    data=csv_sample.getvalue(),
                    file_name="sample_reviews.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    render_app()
