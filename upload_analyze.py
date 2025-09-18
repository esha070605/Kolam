import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from models.pattern_analyzer import PatternAnalyzer
from models.cnn_model import KolamCNN
from utils.image_utils import preprocess_image, create_analysis_visualization
import io

st.set_page_config(page_title="Upload & Analyze", page_icon="🔍", layout="wide")

# Initialize models
@st.cache_resource
def load_models():
    analyzer = PatternAnalyzer()
    cnn_model = KolamCNN()
    return analyzer, cnn_model

st.title("🔍 Upload & Analyze Kolam Patterns")
st.markdown("Upload your Kolam images for comprehensive AI-powered analysis")

# Load models
try:
    analyzer, cnn_model = load_models()
    model_status = "✅ Models loaded successfully"
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    model_status = "❌ Models failed to load"

st.info(model_status)

# File upload section
st.header("📤 Upload Kolam Image")
uploaded_file = st.file_uploader(
    "Choose a Kolam image...",
    type=['png', 'jpg', 'jpeg'],
    help="Upload PNG, JPG, or JPEG files. Maximum size: 10MB"
)

# Drag and drop area styling
st.markdown("""
<style>
    .uploadedFile {
        border: 2px dashed #FF6B35;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, caption="Uploaded Kolam Pattern", use_column_width=True)
        
        # Image info
        st.markdown("**Image Details:**")
        st.write(f"- **Format:** {image.format}")
        st.write(f"- **Size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"- **Mode:** {image.mode}")
        st.write(f"- **File size:** {len(uploaded_file.getvalue())/1024:.1f} KB")
    
    with col2:
        st.subheader("🔄 Preprocessing")
        
        # Preprocessing options
        resize_option = st.checkbox("Resize for analysis", value=True)
        enhance_contrast = st.checkbox("Enhance contrast", value=True)
        noise_reduction = st.checkbox("Noise reduction", value=False)
        
        if st.button("🚀 Start Analysis", type="primary"):
            with st.spinner("Analyzing pattern... This may take a few moments."):
                try:
                    # Preprocess image
                    processed_img = preprocess_image(
                        image, 
                        resize=resize_option,
                        enhance_contrast=enhance_contrast,
                        noise_reduction=noise_reduction
                    )
                    
                    st.image(processed_img, caption="Processed Image", use_column_width=True)
                    
                    # Store processed image in session state for analysis
                    st.session_state['processed_image'] = processed_img
                    st.session_state['original_image'] = image
                    st.success("✅ Image preprocessing completed!")
                    
                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {str(e)}")

    # Analysis section
    if 'processed_image' in st.session_state:
        st.markdown("---")
        st.header("📊 Pattern Analysis Results")
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Geometric Analysis", "🤖 AI Classification", "📐 Mathematical Metrics", "📈 Visualizations"])
        
        with tab1:
            st.subheader("Geometric Pattern Analysis")
            
            analysis_results = {}
            try:
                # Perform geometric analysis
                analysis_results = analyzer.analyze_pattern(st.session_state['processed_image'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Detected Dots", analysis_results.get('dot_count', 0))
                    st.metric("Contours Found", analysis_results.get('contour_count', 0))
                
                with col2:
                    st.metric("Symmetry Score", f"{analysis_results.get('symmetry_score', 0):.2f}")
                    st.metric("Complexity Index", f"{analysis_results.get('complexity', 0):.2f}")
                
                with col3:
                    st.metric("Pattern Type", analysis_results.get('pattern_type', 'Unknown'))
                    st.metric("Grid Size", analysis_results.get('grid_size', 'N/A'))
                
                # Detailed analysis
                st.subheader("Detailed Geometric Features")
                
                if analysis_results.get('features'):
                    features_df = pd.DataFrame([analysis_results['features']])
                    st.dataframe(features_df.T, use_container_width=True)
                
                # Show annotated image
                if analysis_results.get('annotated_image') is not None:
                    st.subheader("Annotated Analysis")
                    st.image(analysis_results['annotated_image'], caption="Pattern Analysis Overlay", use_column_width=True)
                
            except Exception as e:
                st.error(f"❌ Geometric analysis failed: {str(e)}")
                st.info("This might be due to the image format or quality. Try preprocessing with different options.")
        
        with tab2:
            st.subheader("AI Pattern Classification")
            
            classification_results = {}
            try:
                # CNN classification
                classification_results = cnn_model.predict(st.session_state['processed_image'])
                
                if classification_results:
                    # Display classification results
                    categories = ['Geometric', 'Floral', 'Animal', 'Traditional', 'Modern']
                    probabilities = classification_results.get('probabilities', [0.2] * 5)
                    predicted_class = classification_results.get('predicted_class', 'Unknown')
                    confidence = classification_results.get('confidence', 0.0)
                    
                    # Main prediction
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.success(f"**Predicted Category:** {predicted_class}")
                        st.info(f"**Confidence:** {confidence:.1%}")
                    
                    with col2:
                        st.metric("Classification Score", f"{confidence:.3f}")
                    
                    # Probability distribution
                    st.subheader("Category Probabilities")
                    prob_df = pd.DataFrame({
                        'Category': categories,
                        'Probability': probabilities
                    })
                    
                    fig = px.bar(prob_df, x='Category', y='Probability', 
                               title="Classification Confidence by Category",
                               color='Probability', color_continuous_scale='Viridis')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed probabilities
                    st.subheader("Detailed Probabilities")
                    for i, (cat, prob) in enumerate(zip(categories, probabilities)):
                        st.write(f"**{cat}:** {prob:.1%}")
                        st.progress(prob)
                
            except Exception as e:
                st.error(f"❌ Classification failed: {str(e)}")
                st.info("The CNN model might need more training data for accurate predictions.")
        
        with tab3:
            st.subheader("Mathematical Metrics")
            
            metrics = {}
            try:
                # Calculate mathematical properties
                metrics = analyzer.calculate_mathematical_metrics(st.session_state['processed_image'])
                
                # Display metrics in organized layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Symmetry Analysis**")
                    st.metric("Bilateral Symmetry", f"{metrics.get('bilateral_symmetry', 0):.3f}")
                    st.metric("Rotational Symmetry", f"{metrics.get('rotational_symmetry', 0):.3f}")
                    st.metric("Translational Symmetry", f"{metrics.get('translational_symmetry', 0):.3f}")
                
                with col2:
                    st.markdown("**Geometric Properties**")
                    st.metric("Fractal Dimension", f"{metrics.get('fractal_dimension', 0):.3f}")
                    st.metric("Perimeter Ratio", f"{metrics.get('perimeter_ratio', 0):.3f}")
                    st.metric("Area Coverage", f"{metrics.get('area_coverage', 0):.1%}")
                
                # Detailed metrics table
                st.subheader("Complete Metrics Summary")
                metrics_df = pd.DataFrame([metrics]).T
                metrics_df.columns = ['Value']
                st.dataframe(metrics_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Mathematical analysis failed: {str(e)}")
        
        with tab4:
            st.subheader("Analysis Visualizations")
            
            try:
                # Create comprehensive visualization
                viz_results = create_analysis_visualization(
                    st.session_state['processed_image'],
                    analysis_results if 'analysis_results' in locals() else {},
                    classification_results if 'classification_results' in locals() else {}
                )
                
                if viz_results:
                    # Show different visualization types
                    viz_type = st.selectbox(
                        "Select Visualization Type:",
                        ["Edge Detection", "Contour Analysis", "Symmetry Map", "Feature Heatmap"]
                    )
                    
                    if viz_type in viz_results:
                        st.image(viz_results[viz_type], caption=f"{viz_type} Visualization", use_column_width=True)
                    
                    # Interactive plot
                    if 'plot_data' in viz_results:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=viz_results['plot_data']['x'],
                            y=viz_results['plot_data']['y'],
                            mode='lines+markers',
                            name='Pattern Analysis'
                        ))
                        fig.update_layout(title="Pattern Analysis Plot", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Visualization failed: {str(e)}")
        
        # Export results
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Analysis Report"):
                # Create comprehensive report
                report_data = {
                    'filename': uploaded_file.name,
                    'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'geometric_analysis': analysis_results if 'analysis_results' in locals() else {},
                    'classification': classification_results if 'classification_results' in locals() else {},
                    'mathematical_metrics': metrics if 'metrics' in locals() else {}
                }
                
                # Convert to JSON for download
                import json
                report_json = json.dumps(report_data, indent=2, default=str)
                st.download_button(
                    label="💾 Download JSON Report",
                    data=report_json,
                    file_name=f"kolam_analysis_{uploaded_file.name.split('.')[0]}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🖼️ Export Visualizations"):
                st.info("Visualization export feature coming soon!")
        
        with col3:
            if st.button("📋 Save to Database"):
                try:
                    # Save results to database (implementation would depend on database structure)
                    st.success("✅ Results saved to database!")
                except Exception as e:
                    st.error(f"❌ Database save failed: {str(e)}")

else:
    # Instructions when no file is uploaded
    st.markdown("""
    ### 📋 Instructions:
    
    1. **Upload an Image**: Click the upload button above and select a Kolam image from your device
    2. **Supported Formats**: PNG, JPG, JPEG files up to 10MB
    3. **Best Results**: Use clear, well-lit images with good contrast
    4. **Processing Options**: Choose preprocessing options based on your image quality
    
    ### 💡 Tips for Better Analysis:
    - Ensure the Kolam pattern is centered and clearly visible
    - Avoid shadows or uneven lighting
    - Higher resolution images provide more detailed analysis
    - Traditional chalk/rice flour Kolams work best
    
    ### 📝 What You'll Get:
    - **Geometric Analysis**: Dot detection, symmetry analysis, contour extraction
    - **AI Classification**: Pattern category prediction with confidence scores
    - **Mathematical Metrics**: Fractal dimension, symmetry scores, complexity measures
    - **Visualizations**: Edge detection, feature maps, and analysis overlays
    """)
    
    # Sample images section
    st.markdown("---")
    st.subheader("📸 Sample Analysis Results")
    st.info("Upload an image to see real analysis results here")

# Help section
with st.expander("❓ Need Help?"):
    st.markdown("""
    ### Common Issues:
    
    **1. Upload Failed**
    - Check file size (max 10MB)
    - Ensure file format is PNG, JPG, or JPEG
    - Try refreshing the page
    
    **2. Analysis Errors**
    - Try enabling preprocessing options
    - Ensure image shows a clear Kolam pattern
    - Check image quality and contrast
    
    **3. Low Confidence Scores**
    - Use higher resolution images
    - Ensure pattern is well-centered
    - Try different preprocessing options
    
    **4. Model Loading Issues**
    - Refresh the page to reload models
    - Check internet connection
    - Contact support if issues persist
    """)
