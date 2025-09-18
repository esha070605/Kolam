import streamlit as st
import os
from utils.database import init_database

# Page configuration
st.set_page_config(
    page_title="Deep Kolam - AI Pattern Recognition",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_database()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎨 Deep Kolam - AI Pattern Recognition</h1>
    <p>Discover the mathematical beauty of traditional Kolam patterns through AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Welcome to Deep Kolam")
    st.markdown("""
    **Deep Kolam** is an advanced AI-powered application that combines traditional South Indian art with modern computer vision and machine learning technologies. 
    
    ### What is Kolam?
    Kolam is a traditional decorative art form from South India, created using rice flour and geometric patterns. These intricate designs follow mathematical principles and have deep cultural significance.
    
    ### Features of This Application:
    """)
    
    # Feature cards
    st.markdown("""
    <div class="feature-card">
        <h4>🔍 Pattern Analysis</h4>
        <p>Upload Kolam images and get detailed mathematical analysis including symmetry detection, dot grid analysis, and geometric properties.</p>
    </div>
    
    <div class="feature-card">
        <h4>🤖 AI Classification</h4>
        <p>Our CNN model classifies Kolam patterns into categories: Geometric, Floral, Animal, Traditional, and Modern.</p>
    </div>
    
    <div class="feature-card">
        <h4>✨ Pattern Generation</h4>
        <p>Generate new Kolam patterns based on mathematical principles and customizable parameters.</p>
    </div>
    
    <div class="feature-card">
        <h4>🎨 Interactive Drawing</h4>
        <p>Create your own Kolam patterns using our interactive drawing canvas with grid assistance.</p>
    </div>
    
    <div class="feature-card">
        <h4>📚 Learning Mode</h4>
        <p>Explore tutorials, historical context, and mathematical principles behind Kolam art.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.header("Quick Stats")
    
    # Sample metrics (in a real app, these would come from the database)
    st.markdown("""
    <div class="metric-card">
        <h3>156</h3>
        <p>Patterns Analyzed</p>
    </div>
    
    <div class="metric-card">
        <h3>89%</h3>
        <p>Classification Accuracy</p>
    </div>
    
    <div class="metric-card">
        <h3>42</h3>
        <p>Generated Patterns</p>
    </div>
    
    <div class="metric-card">
        <h3>5</h3>
        <p>Pattern Categories</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Recent Activity")
    st.info("🔄 Model training completed")
    st.success("✅ New pattern uploaded: Traditional_Flower.jpg")
    st.info("📊 Weekly analytics generated")

# Navigation instructions
st.markdown("---")
st.markdown("""
### How to Get Started:
1. **Upload & Analyze**: Go to the 'Upload & Analyze' page to upload your Kolam images
2. **Generate Patterns**: Use the 'Generate Kolam' page to create new patterns
3. **Interactive Drawing**: Try the drawing canvas to create patterns manually
4. **Learn More**: Visit the 'Learn Mode' for tutorials and background information

Use the sidebar navigation to explore different features of the application.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Deep Kolam - Preserving tradition through technology</p>
    <p>Built with ❤️ using Streamlit, OpenCV, and TensorFlow</p>
</div>
""", unsafe_allow_html=True)

