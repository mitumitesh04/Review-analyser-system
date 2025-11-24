import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from predict import FakeReviewDetector

# Page configuration
st.set_page_config(
    page_title="Smart Review Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize detector
@st.cache_resource
def load_model():
    return FakeReviewDetector()

try:
    detector = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Header
st.title("🔍 Smart Product Review Analyzer")
st.markdown("**AI-Powered Review Analysis System** | RoBERTa + Multi-Aspect Sentiment Analysis")
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🔍 Deep Analysis", "📊 Bulk Intelligence"])

# TAB 1: Deep Single Review Analysis
with tab1:
    st.subheader("Complete Review Analysis")
    
    # Example reviews
    examples = {
        "Select an example...": "",
        "Mixed Review": "The camera quality is absolutely amazing and takes stunning photos. However, the battery life is terrible and barely lasts half a day. The price is reasonable for what you get.",
        "Positive Review": "Great product! The screen is bright and clear, performance is fast and smooth. Build quality feels premium and solid. Highly recommend!",
        "Negative Review": "Very disappointed. The sound quality is poor and tinny. Delivery took forever, arrived late. Not worth the expensive price tag."
    }
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_example = st.selectbox("Load example review:", list(examples.keys()))
    with col2:
        st.write("")  # Spacing
    
    review_input = st.text_area(
        "Enter product review for analysis:",
        value=examples[selected_example],
        placeholder="Enter a product review to analyze its authenticity and sentiment...",
        height=120
    )
    
    analyze_button = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
    
    if analyze_button:
        if review_input and review_input.strip():
            try:
                with st.spinner("Analyzing review..."):
                    analysis = detector.complete_analysis(review_input)
                
                # Main Metrics
                st.markdown("### Analysis Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    fake = analysis['fake_detection']
                    icon = "🔴" if fake['prediction'] == 'FAKE' else "🟢"
                    st.metric("Authenticity", f"{icon} {fake['prediction']}")
                    st.caption(f"Confidence: {fake['confidence']}%")
                
                with col2:
                    sent = analysis['overall_sentiment']
                    st.metric("Sentiment", f"{sent['emoji']} {sent['sentiment']}")
                    st.caption(f"Rating: {sent['stars']}/5.0 ⭐")
                
                with col3:
                    st.metric("Trust Score", f"{fake['trust_score']}/100")
                    trust_label = "High" if fake['trust_score'] >= 70 else "Medium" if fake['trust_score'] >= 40 else "Low"
                    st.caption(f"Level: {trust_label}")
                
                with col4:
                    aspects_found = len(analysis['aspects'])
                    st.metric("Features Detected", aspects_found)
                    st.caption("Product aspects")
                
                st.markdown("---")
                
                # Sentiment & Trust Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Sentiment Distribution")
                    sent_data = analysis['overall_sentiment']
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Positive', 'Negative', 'Neutral'],
                        values=[sent_data['positive'], sent_data['negative'], sent_data['neutral']],
                        marker_colors=['#4CAF50', '#F44336', '#9E9E9E'],
                        hole=0.4
                    )])
                    fig.update_layout(
                        height=300,
                        showlegend=True,
                        margin=dict(t=30, b=0, l=0, r=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Sentiment Score: {sent_data['compound']} (Range: -1 to +1)")
                
                with col2:
                    st.markdown("#### Trust Score Analysis")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=fake['trust_score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#1f77b4"},
                            'steps': [
                                {'range': [0, 40], 'color': "#ffcdd2"},
                                {'range': [40, 70], 'color': "#fff9c4"},
                                {'range': [70, 100], 'color': "#c8e6c9"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Aspect-Based Analysis
                if analysis['aspects']:
                    st.markdown("---")
                    st.markdown("### Aspect-Based Sentiment Analysis")
                    st.caption("Sentiment analysis for individual product features:")
                    
                    # Create aspect cards
                    aspect_data = []
                    for aspect_name, aspect_info in analysis['aspects'].items():
                        aspect_data.append({
                            'Feature': aspect_name,
                            'Sentiment': aspect_info['sentiment'],
                            'Score': aspect_info['score'],
                            'Emoji': aspect_info['emoji']
                        })
                    
                    # Display aspect metrics
                    num_cols = min(len(aspect_data), 4)
                    cols = st.columns(num_cols)
                    
                    for idx, aspect in enumerate(aspect_data):
                        with cols[idx % num_cols]:
                            st.metric(
                                label=f"{aspect['Emoji']} {aspect['Feature']}",
                                value=aspect['Sentiment'],
                                delta=f"Score: {aspect['Score']}"
                            )
                    
                    # Detailed aspect information
                    with st.expander("📋 View Detailed Aspect Analysis", expanded=False):
                        for aspect_name, aspect_info in analysis['aspects'].items():
                            st.markdown(f"**{aspect_info['emoji']} {aspect_name}** — {aspect_info['sentiment']}")
                            progress_val = (aspect_info['score'] + 1) / 2
                            st.progress(progress_val)
                            st.caption(f"Context: {aspect_info['text']}")
                            st.caption(f"Distribution → Positive: {aspect_info['positive']}% | Negative: {aspect_info['negative']}% | Neutral: {aspect_info['neutral']}%")
                            st.markdown("")
                    
                    # Aspect sentiment visualization
                    if len(aspect_data) > 0:
                        st.markdown("#### Feature Sentiment Comparison")
                        df = pd.DataFrame(aspect_data)
                        fig = px.bar(
                            df,
                            x='Feature',
                            y='Score',
                            color='Score',
                            color_continuous_scale=['#F44336', '#FFC107', '#4CAF50'],
                            title='Sentiment Scores by Product Feature',
                            labels={'Score': 'Sentiment Score'}
                        )
                        fig.update_layout(
                            height=400,
                            xaxis_title="Product Feature",
                            yaxis_title="Sentiment Score (-1 to +1)",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("ℹ️ No specific product features detected. The review does not mention aspects like battery, camera, screen, etc.")
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("⚠️ Please enter a review to analyze.")

# TAB 2: Bulk Analysis
with tab2:
    st.subheader("Multi-Review Product Intelligence")
    st.caption("Analyze multiple reviews to identify trends and aggregate sentiment across product features.")
    
    reviews_input = st.text_area(
        "Enter multiple reviews (one per line):",
        placeholder="Review 1: The camera is great but battery life is poor\nReview 2: Amazing screen quality and fast performance\nReview 3: Terrible delivery experience, arrived damaged",
        height=180
    )
    
    analyze_bulk = st.button("📊 Analyze All Reviews", type="primary", use_container_width=True)
    
    if analyze_bulk:
        if reviews_input and reviews_input.strip():
            reviews_list = [r.strip() for r in reviews_input.split('\n') if r.strip()]
            
            if len(reviews_list) > 0:
                try:
                    with st.spinner(f"Analyzing {len(reviews_list)} reviews..."):
                        result = detector.analyze_reviews(reviews_list)
                    
                    # Summary Metrics
                    st.markdown("### Product Intelligence Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Trust Score", f"{result['product_trust_score']}/100")
                    with col2:
                        st.metric("Total Reviews", result['total_reviews'])
                    with col3:
                        st.metric("Authentic Reviews", result['real_reviews'])
                    with col4:
                        st.metric("Fake Reviews", result['fake_reviews'], delta=f"{result['fake_percentage']}%", delta_color="inverse")
                    
                    st.markdown("---")
                    
                    # Authenticity visualization
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        fig = go.Figure(data=[go.Pie(
                            labels=['Authentic Reviews', 'Fake Reviews'],
                            values=[result['real_reviews'], result['fake_reviews']],
                            marker_colors=['#4CAF50', '#F44336'],
                            hole=0.3
                        )])
                        fig.update_layout(
                            title="Review Authenticity Distribution",
                            height=350,
                            margin=dict(t=50, b=0, l=0, r=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Recommendation box
                        st.markdown("#### Recommendation")
                        if result['product_trust_score'] >= 70:
                            st.success("✅ **TRUSTWORTHY** — This product has reliable and authentic reviews.")
                        elif result['product_trust_score'] >= 40:
                            st.warning("⚠️ **CAUTION** — Mixed authenticity detected. Review carefully before purchase.")
                        else:
                            st.error("🚫 **NOT RECOMMENDED** — High proportion of potentially fake reviews detected.")
                        
                        st.metric("Authenticity Rate", f"{100 - result['fake_percentage']}%")
                        st.caption("Percentage of reviews classified as authentic")
                    
                    # Aspect Summary
                    if result.get('aspect_summary'):
                        st.markdown("---")
                        st.markdown("### Feature-Level Sentiment Analysis")
                        st.caption("Aggregated customer sentiment for each product feature across all reviews")
                        
                        # Create summary dataframe
                        aspect_df = pd.DataFrame([
                            {
                                'Feature': aspect,
                                'Overall Sentiment': data['sentiment'],
                                'Avg Score': data['avg_score'],
                                'Mentions': data['mentions'],
                                'Coverage': f"{data['percentage']}%"
                            }
                            for aspect, data in result['aspect_summary'].items()
                        ]).sort_values('Mentions', ascending=False)
                        
                        # Display table
                        st.dataframe(aspect_df, use_container_width=True, hide_index=True)
                        
                        # Visualizations
                        top_aspects = aspect_df.head(8)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Most Discussed Features")
                            fig = px.bar(
                                top_aspects,
                                x='Feature',
                                y='Mentions',
                                color='Mentions',
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(
                                height=350,
                                xaxis_title="Product Feature",
                                yaxis_title="Number of Mentions",
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### Feature Sentiment Scores")
                            fig = px.bar(
                                top_aspects,
                                x='Feature',
                                y='Avg Score',
                                color='Avg Score',
                                color_continuous_scale=['#F44336', '#FFC107', '#4CAF50'],
                                labels={'Avg Score': 'Sentiment'}
                            )
                            fig.update_layout(
                                height=350,
                                xaxis_title="Product Feature",
                                yaxis_title="Average Sentiment Score",
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Key insights
                        st.markdown("---")
                        st.markdown("### Key Insights")
                        
                        positive_aspects = [k for k, v in result['aspect_summary'].items() if v['avg_score'] > 0.1]
                        negative_aspects = [k for k, v in result['aspect_summary'].items() if v['avg_score'] < -0.1]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if positive_aspects:
                                st.success(f"**Product Strengths:** {', '.join(positive_aspects)}")
                            else:
                                st.info("**Product Strengths:** No clear strengths identified")
                        
                        with col2:
                            if negative_aspects:
                                st.error(f"**Areas for Improvement:** {', '.join(negative_aspects)}")
                            else:
                                st.info("**Areas for Improvement:** No major weaknesses identified")
                    else:
                        st.info("ℹ️ No product features were mentioned across the reviews.")
                
                except Exception as e:
                    st.error(f"Error during bulk analysis: {str(e)}")
            else:
                st.warning("⚠️ No valid reviews found. Please enter at least one review.")
        else:
            st.warning("⚠️ Please enter reviews to analyze.")

# Professional Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    
    st.markdown("## About This System")
    st.markdown("""
    **Smart Review Analyzer** is an AI-powered system designed to detect fake reviews 
    and perform multi-aspect sentiment analysis on product reviews.
    """)
    
    st.markdown("---")
    
    st.markdown("### Core Technologies")
    st.markdown("""
    - **RoBERTa Model**: 125M parameter transformer for fake detection
    - **VADER Sentiment**: Lexicon-based sentiment analysis
    - **Aspect Extraction**: Custom NLP pipeline for feature detection
    """)
    
    st.markdown("---")
    
    st.markdown("### Key Capabilities")
    st.markdown("""
    ✓ Fake review detection  
    ✓ Overall sentiment analysis  
    ✓ Aspect-based sentiment  
    ✓ Multi-review aggregation  
    ✓ Trust score calculation
    """)
    
    st.markdown("---")
    
    st.markdown("### Detected Aspects")
    st.caption("The system automatically identifies:")
    st.code("""
- Battery
- Camera
- Screen
- Price
- Quality
- Delivery
- Design
- Performance
- Size
- Sound
    """)
    
    st.markdown("---")
    
    st.markdown("### Model Performance")
    st.info("""
    **Accuracy**: 85-92%  
    **Training Data**: 40,000+ reviews  
    **Model Type**: Fine-tuned RoBERTa
    """)
    
    st.markdown("---")
    
    st.caption("Developed using Transformers, PyTorch, and Streamlit")