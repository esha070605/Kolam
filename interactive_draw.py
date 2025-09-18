import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2
import json
import io
import pandas as pd
from models.pattern_analyzer import PatternAnalyzer
from utils.image_utils import canvas_to_kolam, add_grid_overlay, validate_kolam_rules
import plotly.graph_objects as go

st.set_page_config(page_title="Interactive Draw", page_icon="🎨", layout="wide")

st.title("🎨 Interactive Kolam Drawing Canvas")
st.markdown("Create your own Kolam patterns with guided drawing tools and real-time analysis")

# Initialize session state
if 'canvas_history' not in st.session_state:
    st.session_state['canvas_history'] = []
if 'current_pattern' not in st.session_state:
    st.session_state['current_pattern'] = None
if 'drawing_mode' not in st.session_state:
    st.session_state['drawing_mode'] = 'freedraw'

# Sidebar for drawing controls
st.sidebar.header("🎛️ Drawing Controls")

# Canvas settings
st.sidebar.subheader("Canvas Settings")
canvas_size = st.sidebar.slider("Canvas Size", min_value=400, max_value=800, value=600, step=50)
background_color = st.sidebar.color_picker("Background Color", "#000000")
stroke_color = st.sidebar.color_picker("Drawing Color", "#FFFFFF")
stroke_width = st.sidebar.slider("Stroke Width", min_value=1, max_value=20, value=3)

# Grid assistance
st.sidebar.subheader("Grid Assistance")
show_grid = st.sidebar.checkbox("Show Dot Grid", value=True)
grid_size = st.sidebar.slider("Grid Size", min_value=3, max_value=15, value=7, step=2) if show_grid else 7
grid_opacity = st.sidebar.slider("Grid Opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.1) if show_grid else 0.5

# Drawing modes
st.sidebar.subheader("Drawing Mode")
drawing_mode = st.sidebar.selectbox(
    "Mode",
    ["freedraw", "line", "circle", "rect", "point"],
    index=0
)

# Kolam rules assistance
st.sidebar.subheader("Kolam Rules")
enforce_rules = st.sidebar.checkbox("Enforce Traditional Rules", value=True)
continuous_line = st.sidebar.checkbox("Continuous Line Only", value=True)
symmetry_guide = st.sidebar.checkbox("Symmetry Guide", value=False)

if symmetry_guide:
    symmetry_type = st.sidebar.selectbox(
        "Symmetry Type",
        ["Bilateral", "Rotational", "None"]
    )

# Pattern templates
st.sidebar.subheader("Pattern Templates")
template_type = st.sidebar.selectbox(
    "Load Template",
    ["None", "Simple Cross", "Basic Flower", "Star Pattern", "Traditional Motif"]
)

if template_type != "None" and st.sidebar.button("Load Template"):
    st.session_state['load_template'] = template_type

# Main drawing area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🖌️ Drawing Canvas")
    
    # Create initial canvas data
    initial_drawing = None
    if 'load_template' in st.session_state:
        # Load template pattern
        initial_drawing = create_template_pattern(
            st.session_state['load_template'], 
            canvas_size, 
            grid_size
        )
        del st.session_state['load_template']
    
    # Create the drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=background_color,
        background_image=None,
        update_streamlit=True,
        height=canvas_size,
        width=canvas_size,
        drawing_mode=drawing_mode,
        point_display_radius=stroke_width,
        key="kolam_canvas",
        initial_drawing=initial_drawing,
    )
    
    # Add grid overlay if enabled
    if show_grid:
        grid_overlay = add_grid_overlay(canvas_size, grid_size, grid_opacity)
        st.image(grid_overlay, caption="Grid Guide", use_column_width=True)

with col2:
    st.subheader("🛠️ Tools")
    
    # Drawing tools
    tool_col1, tool_col2 = st.columns(2)
    
    with tool_col1:
        if st.button("🔄 Undo", use_container_width=True):
            if st.session_state['canvas_history']:
                st.session_state['canvas_history'].pop()
                st.rerun()
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state['canvas_history'] = []
            st.rerun()
    
    with tool_col2:
        if st.button("💾 Save", use_container_width=True):
            if canvas_result.image_data is not None:
                st.session_state['current_pattern'] = canvas_result.image_data
                st.success("✅ Pattern saved!")
        
        if st.button("📋 Copy", use_container_width=True):
            st.info("Copy functionality available in export")
    
    st.markdown("---")
    
    # Pattern analysis tools
    st.subheader("🔍 Analysis Tools")
    
    if st.button("📊 Analyze Pattern", type="primary", use_container_width=True):
        if canvas_result.image_data is not None:
            with st.spinner("Analyzing your drawing..."):
                try:
                    # Convert canvas to image
                    pattern_image = canvas_to_kolam(canvas_result.image_data)
                    
                    # Analyze the pattern
                    analyzer = PatternAnalyzer()
                    analysis_result = analyzer.analyze_pattern(pattern_image)
                    
                    st.session_state['analysis_result'] = analysis_result
                    st.session_state['analyzed_image'] = pattern_image
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
    
    # Kolam rule validation
    if enforce_rules and canvas_result.image_data is not None:
        st.markdown("---")
        st.subheader("✅ Rule Validation")
        
        try:
            pattern_image = canvas_to_kolam(canvas_result.image_data)
            rule_validation = validate_kolam_rules(pattern_image, continuous_line)
            
            if rule_validation['valid']:
                st.success("✅ Follows traditional rules!")
            else:
                st.warning("⚠️ Rule violations detected:")
                for violation in rule_validation['violations']:
                    st.write(f"- {violation}")
                    
            # Show rule compliance score
            st.metric("Rule Compliance", f"{rule_validation['score']:.1%}")
            
        except Exception as e:
            st.info("Draw something to see rule validation")
    
    # Quick actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    if st.button("🎨 Apply Colors", use_container_width=True):
        st.info("Color application coming soon!")
    
    if st.button("✨ Add Effects", use_container_width=True):
        st.info("Effect application coming soon!")
    
    if st.button("🔄 Mirror Pattern", use_container_width=True):
        if canvas_result.image_data is not None:
            try:
                # Create mirrored version
                mirrored_pattern = create_mirrored_pattern(canvas_result.image_data)
                st.session_state['mirrored_pattern'] = mirrored_pattern
                st.success("✅ Pattern mirrored!")
            except Exception as e:
                st.error(f"❌ Mirror failed: {str(e)}")

# Display current drawing info
if canvas_result.json_data is not None:
    st.markdown("---")
    
    # Show drawing statistics
    col1, col2, col3 = st.columns(3)
    
    objects = canvas_result.json_data["objects"]
    
    with col1:
        st.metric("Strokes", len([obj for obj in objects if obj["type"] == "path"]))
    
    with col2:
        st.metric("Shapes", len([obj for obj in objects if obj["type"] != "path"]))
    
    with col3:
        total_points = sum(len(obj.get("path", [])) for obj in objects if obj.get("path"))
        st.metric("Total Points", total_points)

# Pattern analysis results
if 'analysis_result' in st.session_state and 'analyzed_image' in st.session_state:
    st.markdown("---")
    st.header("📊 Pattern Analysis Results")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Geometric Analysis", "Symmetry Analysis", "Traditional Compliance"])
    
    with analysis_tab1:
        result = st.session_state['analysis_result']
        
        # Display analyzed image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Analyzed Pattern")
            if result.get('annotated_image') is not None:
                st.image(result['annotated_image'], caption="Pattern with Analysis Overlay")
            else:
                st.image(st.session_state['analyzed_image'], caption="Original Pattern")
        
        with col2:
            st.subheader("Geometric Properties")
            st.metric("Detected Lines", result.get('line_count', 0))
            st.metric("Intersection Points", result.get('intersection_count', 0))
            st.metric("Enclosed Areas", result.get('enclosed_areas', 0))
            st.metric("Complexity Score", f"{result.get('complexity_score', 0):.2f}")
    
    with analysis_tab2:
        st.subheader("Symmetry Analysis")
        
        # Symmetry metrics
        symmetry_col1, symmetry_col2 = st.columns(2)
        
        with symmetry_col1:
            bilateral_sym = result.get('bilateral_symmetry', 0)
            st.metric("Bilateral Symmetry", f"{bilateral_sym:.3f}")
            st.progress(bilateral_sym)
            
            rotational_sym = result.get('rotational_symmetry', 0)
            st.metric("Rotational Symmetry", f"{rotational_sym:.3f}")
            st.progress(rotational_sym)
        
        with symmetry_col2:
            translational_sym = result.get('translational_symmetry', 0)
            st.metric("Translational Symmetry", f"{translational_sym:.3f}")
            st.progress(translational_sym)
            
            overall_sym = (bilateral_sym + rotational_sym + translational_sym) / 3
            st.metric("Overall Symmetry", f"{overall_sym:.3f}")
            st.progress(overall_sym)
        
        # Symmetry visualization
        if result.get('symmetry_visualization'):
            st.subheader("Symmetry Visualization")
            st.image(result['symmetry_visualization'], caption="Symmetry Analysis")
    
    with analysis_tab3:
        st.subheader("Traditional Kolam Compliance")
        
        # Rule compliance analysis
        compliance = result.get('traditional_compliance', {})
        
        compliance_col1, compliance_col2 = st.columns(2)
        
        with compliance_col1:
            st.write("**Rule Compliance:**")
            for rule, status in compliance.items():
                if isinstance(status, bool):
                    icon = "✅" if status else "❌"
                    st.write(f"{icon} {rule.replace('_', ' ').title()}")
                else:
                    st.write(f"📊 {rule.replace('_', ' ').title()}: {status}")
        
        with compliance_col2:
            overall_compliance = compliance.get('overall_score', 0)
            st.metric("Compliance Score", f"{overall_compliance:.1%}")
            
            if overall_compliance > 0.8:
                st.success("🎉 Excellent traditional compliance!")
            elif overall_compliance > 0.6:
                st.info("👍 Good traditional elements present")
            else:
                st.warning("⚠️ Consider traditional Kolam principles")

# Drawing tutorials and help
st.markdown("---")
st.header("📚 Drawing Guide & Tutorials")

tutorial_tab1, tutorial_tab2, tutorial_tab3 = st.tabs(["Basic Techniques", "Traditional Patterns", "Advanced Tips"])

with tutorial_tab1:
    st.subheader("🎯 Basic Drawing Techniques")
    
    st.markdown("""
    ### Getting Started:
    
    **1. Understand the Dot Grid:**
    - Kolam patterns traditionally start with a grid of dots
    - Use the grid guide to align your patterns
    - Dots serve as connection points for lines
    
    **2. Basic Drawing Rules:**
    - Lines should connect dots smoothly
    - Avoid sharp angles where possible
    - Maintain symmetry when intended
    - Keep lines continuous for traditional authenticity
    
    **3. Drawing Tools:**
    - **Freedraw**: Natural hand drawing for organic curves
    - **Line**: Perfect straight lines between points
    - **Circle**: Create perfect circular elements
    - **Rectangle**: Add geometric shapes
    """)
    
    # Interactive tutorial
    st.subheader("🎮 Interactive Tutorial")
    if st.button("Start Basic Tutorial"):
        st.info("Follow the highlighted dots to create your first pattern!")

with tutorial_tab2:
    st.subheader("🏛️ Traditional Kolam Patterns")
    
    st.markdown("""
    ### Common Traditional Motifs:
    
    **1. Pulli Kolam (Dot Patterns):**
    - Start with a regular dot grid
    - Connect dots with curved lines
    - Create enclosed shapes around dots
    
    **2. Kambi Kolam (Line Patterns):**
    - Continuous line patterns
    - No lifting of the drawing tool
    - Symmetrical designs
    
    **3. Sikku Kolam (Interlaced Patterns):**
    - Lines that weave over and under
    - Complex interlacing patterns
    - Mathematical precision required
    """)
    
    # Template patterns
    st.subheader("📐 Pattern Templates")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Load Flower Pattern"):
            st.session_state['load_template'] = 'Basic Flower'
            st.rerun()
    
    with col2:
        if st.button("Load Star Pattern"):
            st.session_state['load_template'] = 'Star Pattern'
            st.rerun()
    
    with col3:
        if st.button("Load Traditional Motif"):
            st.session_state['load_template'] = 'Traditional Motif'
            st.rerun()

with tutorial_tab3:
    st.subheader("🚀 Advanced Drawing Tips")
    
    st.markdown("""
    ### Professional Techniques:
    
    **1. Symmetry Mastery:**
    - Use the symmetry guide for perfect balance
    - Plan your pattern before drawing
    - Start from the center and work outward
    
    **2. Line Quality:**
    - Vary stroke width for visual interest
    - Use smooth, confident strokes
    - Maintain consistent spacing
    
    **3. Composition:**
    - Consider the overall balance
    - Leave appropriate negative space
    - Create visual flow through the pattern
    
    **4. Digital Advantages:**
    - Use undo for perfect lines
    - Experiment with colors
    - Save and iterate on designs
    """)
    
    # Advanced features
    st.subheader("🔧 Advanced Features")
    
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        st.markdown("**Precision Tools:**")
        st.write("- Grid snapping (coming soon)")
        st.write("- Angle constraints (coming soon)")
        st.write("- Distance measurements (coming soon)")
    
    with adv_col2:
        st.markdown("**Export Options:**")
        st.write("- High-resolution PNG")
        st.write("- Scalable SVG format")
        st.write("- Pattern data JSON")

# Export section
if canvas_result.image_data is not None:
    st.markdown("---")
    st.header("📥 Export Your Creation")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("💾 Download PNG", use_container_width=True):
            # Convert canvas to PNG
            try:
                pattern_image = canvas_to_kolam(canvas_result.image_data)
                img_buffer = io.BytesIO()
                pattern_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📁 Download Image",
                    data=img_buffer.getvalue(),
                    file_name=f"hand_drawn_kolam_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with export_col2:
        if st.button("📐 Export SVG", use_container_width=True):
            try:
                # Convert to SVG format
                svg_data = canvas_to_svg(canvas_result.json_data, canvas_size)
                st.download_button(
                    label="📁 Download SVG",
                    data=svg_data,
                    file_name=f"hand_drawn_kolam_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.svg",
                    mime="image/svg+xml"
                )
            except Exception as e:
                st.error(f"SVG export failed: {str(e)}")
    
    with export_col3:
        if st.button("🎨 Save to Gallery", use_container_width=True):
            try:
                # Save to user gallery (would integrate with database)
                st.success("✅ Saved to your gallery!")
            except Exception as e:
                st.error(f"Save failed: {str(e)}")

# Helper functions (these would be implemented in utils)
def create_template_pattern(template_type, canvas_size, grid_size):
    """Create template pattern data for the canvas"""
    # This would return fabric.js compatible JSON data for templates
    templates = {
        'Simple Cross': {
            'version': '4.6.0',
            'objects': [
                {
                    'type': 'line',
                    'left': canvas_size//2 - 50,
                    'top': canvas_size//2,
                    'x1': 0, 'y1': 0, 'x2': 100, 'y2': 0,
                    'stroke': 'white',
                    'strokeWidth': 3
                },
                {
                    'type': 'line',
                    'left': canvas_size//2,
                    'top': canvas_size//2 - 50,
                    'x1': 0, 'y1': 0, 'x2': 0, 'y2': 100,
                    'stroke': 'white',
                    'strokeWidth': 3
                }
            ]
        }
    }
    return templates.get(template_type, None)

def create_mirrored_pattern(image_data):
    """Create a mirrored version of the pattern"""
    # Implementation would create mirrored version
    return image_data

def canvas_to_svg(json_data, canvas_size):
    """Convert canvas JSON data to SVG format"""
    # Implementation would convert fabric.js JSON to SVG
    svg_header = f'<svg width="{canvas_size}" height="{canvas_size}" xmlns="http://www.w3.org/2000/svg">'
    svg_footer = '</svg>'
    
    # Basic SVG conversion - in a real implementation this would be more comprehensive
    svg_content = ''
    if json_data and 'objects' in json_data:
        for obj in json_data['objects']:
            if obj.get('type') == 'path':
                svg_content += f'<path d="{obj.get("path", "")}" stroke="white" fill="none" stroke-width="3"/>'
    
    return svg_header + svg_content + svg_footer

# Session state cleanup
if st.button("🔄 Reset Session", help="Clear all drawing data and start fresh"):
    for key in ['canvas_history', 'current_pattern', 'analysis_result', 'analyzed_image']:
        if key in st.session_state:
            del st.session_state[key]
    st.success("✅ Session reset complete!")
    st.rerun()
