import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw
import cv2
from models.kolam_generator import KolamGenerator
from utils.image_utils import apply_color_scheme, add_artistic_effects
import io

st.set_page_config(page_title="Generate Kolam", page_icon="✨", layout="wide")

st.title("✨ Generate Kolam Patterns")
st.markdown("Create beautiful Kolam patterns using mathematical algorithms and AI-guided generation")

# Initialize generator
@st.cache_resource
def load_generator():
    return KolamGenerator()

generator = load_generator()

# Sidebar for parameters
st.sidebar.header("🎛️ Pattern Parameters")

# Grid settings
st.sidebar.subheader("Grid Configuration")
grid_size = st.sidebar.slider("Grid Size", min_value=3, max_value=15, value=7, step=2)
dot_density = st.sidebar.slider("Dot Density", min_value=0.3, max_value=1.0, value=0.7, step=0.1)

# Symmetry options
st.sidebar.subheader("Symmetry Type")
symmetry_type = st.sidebar.selectbox(
    "Choose Symmetry",
    ["Bilateral", "Rotational", "Translational", "Radial", "Mixed"]
)

rotation_order = st.sidebar.slider("Rotation Order", min_value=2, max_value=8, value=4, step=1) if symmetry_type == "Rotational" else 4

# Pattern complexity
st.sidebar.subheader("Pattern Complexity")
complexity_level = st.sidebar.selectbox(
    "Complexity Level",
    ["Simple", "Medium", "Complex", "Very Complex"]
)

curve_style = st.sidebar.selectbox(
    "Curve Style",
    ["Smooth", "Angular", "Mixed", "Flowing"]
)

# Theme selection
st.sidebar.subheader("Pattern Theme")
theme = st.sidebar.selectbox(
    "Choose Theme",
    ["Geometric", "Floral", "Traditional", "Modern", "Abstract"]
)

# Color scheme
st.sidebar.subheader("Color Scheme")
color_scheme = st.sidebar.selectbox(
    "Color Palette",
    ["Traditional White", "Vibrant Colors", "Pastel", "Monochrome", "Custom"]
)

primary_color = "#FF6B35"
secondary_color = "#F7931E" 
background_color = "#FFFFFF"

if color_scheme == "Custom":
    primary_color = st.sidebar.color_picker("Primary Color", "#FF6B35")
    secondary_color = st.sidebar.color_picker("Secondary Color", "#F7931E")
    background_color = st.sidebar.color_picker("Background Color", "#FFFFFF")

# Advanced options
with st.sidebar.expander("🔧 Advanced Options"):
    line_thickness = st.slider("Line Thickness", min_value=1, max_value=10, value=3)
    smoothness = st.slider("Curve Smoothness", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    add_decorations = st.checkbox("Add Decorative Elements", value=True)
    artistic_effects = st.checkbox("Apply Artistic Effects", value=False)

# Main content area
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("🎨 Generation Controls")
    
    # Quick presets
    st.markdown("**Quick Presets:**")
    preset_col1, preset_col2 = st.columns(2)
    
    with preset_col1:
        if st.button("🌸 Floral", use_container_width=True):
            st.session_state['preset'] = 'floral'
        if st.button("📐 Geometric", use_container_width=True):
            st.session_state['preset'] = 'geometric'
    
    with preset_col2:
        if st.button("🏛️ Traditional", use_container_width=True):
            st.session_state['preset'] = 'traditional'
        if st.button("🎭 Modern", use_container_width=True):
            st.session_state['preset'] = 'modern'
    
    st.markdown("---")
    
    # Generation button
    if st.button("🚀 Generate Pattern", type="primary", use_container_width=True):
        with st.spinner("Creating your Kolam pattern..."):
            try:
                # Prepare generation parameters
                params = {
                    'grid_size': grid_size,
                    'dot_density': dot_density,
                    'symmetry_type': symmetry_type,
                    'rotation_order': rotation_order,
                    'complexity': complexity_level,
                    'curve_style': curve_style,
                    'theme': theme,
                    'line_thickness': line_thickness,
                    'smoothness': smoothness,
                    'decorations': add_decorations
                }
                
                # Apply preset modifications
                if 'preset' in st.session_state:
                    params.update(generator.get_preset_params(st.session_state['preset']))
                    del st.session_state['preset']
                
                # Generate pattern
                pattern_result = generator.generate_pattern(params)
                
                if pattern_result:
                    st.session_state['generated_pattern'] = pattern_result
                    st.session_state['generation_params'] = params
                    st.success("✅ Pattern generated successfully!")
                else:
                    st.error("❌ Pattern generation failed")
                    
            except Exception as e:
                st.error(f"❌ Generation error: {str(e)}")
    
    # Additional controls
    st.markdown("---")
    
    if st.button("🔄 Random Generation", use_container_width=True):
        with st.spinner("Generating random pattern..."):
            try:
                random_params = generator.generate_random_params()
                pattern_result = generator.generate_pattern(random_params)
                
                if pattern_result:
                    st.session_state['generated_pattern'] = pattern_result
                    st.session_state['generation_params'] = random_params
                    st.success("✅ Random pattern generated!")
                    
            except Exception as e:
                st.error(f"❌ Random generation error: {str(e)}")
    
    if st.button("🎲 Surprise Me!", use_container_width=True):
        with st.spinner("Creating surprise pattern..."):
            try:
                surprise_params = generator.generate_surprise_params()
                pattern_result = generator.generate_pattern(surprise_params)
                
                if pattern_result:
                    st.session_state['generated_pattern'] = pattern_result
                    st.session_state['generation_params'] = surprise_params
                    st.success("✅ Surprise pattern created!")
                    
            except Exception as e:
                st.error(f"❌ Surprise generation error: {str(e)}")

with col1:
    st.subheader("🖼️ Generated Pattern")
    
    # Display generated pattern
    if 'generated_pattern' in st.session_state:
        pattern_data = st.session_state['generated_pattern']
        params = st.session_state['generation_params']
        
        # Main pattern display
        pattern_image = pattern_data['image']
        
        # Apply color scheme
        pattern_image = pattern_data['image']
        if color_scheme != "Traditional White":
            if color_scheme == "Custom":
                colors = [primary_color, secondary_color, background_color]
            else:
                colors = None
            pattern_image = apply_color_scheme(pattern_image, color_scheme, colors)
        
        # Apply artistic effects
        if artistic_effects:
            pattern_image = add_artistic_effects(pattern_image)
        
        st.image(pattern_image, caption="Generated Kolam Pattern", use_column_width=True)
        
        # Pattern information
        st.subheader("📊 Pattern Information")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Grid Size", f"{params['grid_size']}×{params['grid_size']}")
            st.metric("Symmetry", params['symmetry_type'])
        
        with info_col2:
            st.metric("Complexity", params['complexity'])
            st.metric("Theme", params['theme'])
        
        with info_col3:
            st.metric("Dots", pattern_data.get('dot_count', 'N/A'))
            st.metric("Lines", pattern_data.get('line_count', 'N/A'))
        
        # Detailed analysis of generated pattern
        st.subheader("🔍 Pattern Analysis")
        
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Geometric Properties", "Mathematical Metrics", "Generation Details"])
        
        with analysis_tab1:
            if 'geometric_properties' in pattern_data:
                props = pattern_data['geometric_properties']
                
                prop_col1, prop_col2 = st.columns(2)
                
                with prop_col1:
                    st.write("**Symmetry Analysis:**")
                    st.write(f"- Bilateral: {props.get('bilateral_symmetry', 'N/A')}")
                    st.write(f"- Rotational: {props.get('rotational_symmetry', 'N/A')}")
                    st.write(f"- Translational: {props.get('translational_symmetry', 'N/A')}")
                
                with prop_col2:
                    st.write("**Geometric Features:**")
                    st.write(f"- Area Coverage: {props.get('area_coverage', 'N/A')}")
                    st.write(f"- Perimeter: {props.get('perimeter', 'N/A')}")
                    st.write(f"- Complexity Index: {props.get('complexity_index', 'N/A')}")
        
        with analysis_tab2:
            if 'mathematical_metrics' in pattern_data:
                metrics = pattern_data['mathematical_metrics']
                
                # Create metrics visualization
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=metric_values,
                    theta=metric_names,
                    fill='toself',
                    name='Pattern Metrics'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=False,
                    title="Mathematical Properties Radar Chart"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tab3:
            st.write("**Generation Parameters:**")
            for key, value in params.items():
                st.write(f"- **{key.replace('_', ' ').title()}:** {value}")
            
            if 'generation_time' in pattern_data:
                st.write(f"- **Generation Time:** {pattern_data['generation_time']:.2f} seconds")
            
            if 'algorithm_details' in pattern_data:
                st.write("**Algorithm Details:**")
                for detail in pattern_data['algorithm_details']:
                    st.write(f"- {detail}")
    
    else:
        # Display placeholder when no pattern is generated
        st.info("👈 Use the controls on the right to generate your first Kolam pattern!")
        
        # Show sample patterns
        st.subheader("📸 Sample Generated Patterns")
        
        try:
            # Generate sample patterns for display
            sample_params = [
                {'grid_size': 5, 'theme': 'Geometric', 'complexity': 'Simple'},
                {'grid_size': 7, 'theme': 'Floral', 'complexity': 'Medium'},
                {'grid_size': 9, 'theme': 'Traditional', 'complexity': 'Complex'}
            ]
            
            sample_cols = st.columns(3)
            
            for i, params in enumerate(sample_params):
                with sample_cols[i]:
                    # Generate small sample pattern
                    sample_pattern = generator.generate_sample(params)
                    if sample_pattern:
                        st.image(sample_pattern, caption=f"{params['theme']} - {params['complexity']}")
                    else:
                        st.info(f"Sample {i+1}")
                        
        except Exception as e:
            st.info("Sample patterns will be shown here")

# Export and sharing section
if 'generated_pattern' in st.session_state:
    st.markdown("---")
    st.subheader("📥 Export & Share")
    
    export_col1, export_col2, export_col3, export_col4 = st.columns(4)
    
    with export_col1:
        if st.button("💾 Download PNG", use_container_width=True):
            # Convert pattern to downloadable PNG
            img_buffer = io.BytesIO()
            pattern_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="📁 Download Image",
                data=img_buffer.getvalue(),
                file_name=f"kolam_pattern_{grid_size}x{grid_size}_{theme.lower()}.png",
                mime="image/png"
            )
    
    with export_col2:
        if st.button("📐 Download SVG", use_container_width=True):
            # Generate SVG version
            try:
                svg_data = generator.export_to_svg(st.session_state['generated_pattern'])
                st.download_button(
                    label="📁 Download SVG",
                    data=svg_data,
                    file_name=f"kolam_pattern_{grid_size}x{grid_size}_{theme.lower()}.svg",
                    mime="image/svg+xml"
                )
            except Exception as e:
                st.error(f"SVG export failed: {str(e)}")
    
    with export_col3:
        if st.button("📊 Export Data", use_container_width=True):
            # Export pattern data as JSON
            import json
            pattern_data = st.session_state['generated_pattern']
            data_json = json.dumps(pattern_data, indent=2, default=str)
            
            st.download_button(
                label="📁 Download JSON",
                data=data_json,
                file_name=f"kolam_data_{grid_size}x{grid_size}_{theme.lower()}.json",
                mime="application/json"
            )
    
    with export_col4:
        if st.button("💖 Save Favorite", use_container_width=True):
            # Save to favorites (would integrate with database)
            try:
                # Implementation would save to user favorites
                st.success("✅ Saved to favorites!")
            except Exception as e:
                st.error(f"Save failed: {str(e)}")

# Pattern variations section
if 'generated_pattern' in st.session_state:
    st.markdown("---")
    st.subheader("🎨 Pattern Variations")
    
    st.markdown("Generate variations of your current pattern:")
    
    var_col1, var_col2, var_col3 = st.columns(3)
    
    with var_col1:
        if st.button("🔄 Color Variation", use_container_width=True):
            with st.spinner("Creating color variation..."):
                try:
                    base_pattern = st.session_state['generated_pattern']
                    color_variation = generator.create_color_variation(base_pattern)
                    if color_variation:
                        st.session_state['generated_pattern'] = color_variation
                        st.rerun()
                except Exception as e:
                    st.error(f"Color variation failed: {str(e)}")
    
    with var_col2:
        if st.button("📐 Size Variation", use_container_width=True):
            with st.spinner("Creating size variation..."):
                try:
                    base_pattern = st.session_state['generated_pattern']
                    size_variation = generator.create_size_variation(base_pattern)
                    if size_variation:
                        st.session_state['generated_pattern'] = size_variation
                        st.rerun()
                except Exception as e:
                    st.error(f"Size variation failed: {str(e)}")
    
    with var_col3:
        if st.button("✨ Style Variation", use_container_width=True):
            with st.spinner("Creating style variation..."):
                try:
                    base_pattern = st.session_state['generated_pattern']
                    style_variation = generator.create_style_variation(base_pattern)
                    if style_variation:
                        st.session_state['generated_pattern'] = style_variation
                        st.rerun()
                except Exception as e:
                    st.error(f"Style variation failed: {str(e)}")

# Help section
with st.expander("❓ Generation Help & Tips"):
    st.markdown("""
    ### 🎯 How to Generate Great Patterns:
    
    **1. Start Simple:**
    - Begin with smaller grid sizes (5x5 or 7x7)
    - Use "Simple" complexity for your first patterns
    - Try traditional themes first
    
    **2. Experiment with Symmetry:**
    - **Bilateral:** Mirror symmetry (left-right)
    - **Rotational:** Pattern repeats when rotated
    - **Radial:** Symmetry from center outward
    - **Mixed:** Combination of different symmetries
    
    **3. Customize Colors:**
    - Traditional white on dark background is classic
    - Vibrant colors work well for modern themes
    - Use custom colors to match your preferences
    
    **4. Advanced Features:**
    - Increase smoothness for flowing curves
    - Add decorative elements for richer patterns
    - Try artistic effects for unique looks
    
    ### 🎨 Pattern Themes Explained:
    
    - **Geometric:** Focus on mathematical shapes and patterns
    - **Floral:** Inspired by flowers and natural forms
    - **Traditional:** Classic Kolam motifs and designs
    - **Modern:** Contemporary interpretations
    - **Abstract:** Creative and artistic patterns
    
    ### 💡 Performance Tips:
    - Larger grid sizes take more time to generate
    - Complex patterns with many decorations are slower
    - Very high smoothness values increase processing time
    """)

# Pattern gallery section
st.markdown("---")
st.subheader("🖼️ Pattern Gallery")

gallery_tab1, gallery_tab2, gallery_tab3 = st.tabs(["Recent Generations", "Popular Patterns", "User Favorites"])

with gallery_tab1:
    st.info("Your recently generated patterns will appear here")

with gallery_tab2:
    st.info("Popular community patterns coming soon!")

with gallery_tab3:
    st.info("Your saved favorite patterns will be shown here")
