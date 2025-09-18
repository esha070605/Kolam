import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw
import pandas as pd

st.set_page_config(page_title="Learn Mode", page_icon="📚", layout="wide")

st.title("📚 Learn About Kolam - Interactive Educational Mode")
st.markdown("Discover the rich history, mathematical beauty, and cultural significance of Kolam art")

# Sidebar for navigation
st.sidebar.header("📖 Learning Sections")
section = st.sidebar.radio(
    "Choose Learning Topic:",
    [
        "🏛️ History & Culture",
        "🔢 Mathematics in Kolam",
        "🎨 Types of Kolam",
        "📐 Geometric Principles",
        "🧮 Pattern Analysis",
        "🎯 Step-by-Step Tutorials",
        "🌟 Advanced Concepts",
        "🎮 Interactive Quiz"
    ]
)

if section == "🏛️ History & Culture":
    st.header("🏛️ History & Cultural Significance of Kolam")
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Origins", "Cultural Meaning", "Regional Variations", "Modern Evolution"])
    
    with tab1:
        st.subheader("📜 Ancient Origins")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### The Birth of Kolam Art
            
            Kolam is an ancient art form that originated in **Tamil Nadu, South India**, over 5000 years ago. 
            This traditional practice involves creating intricate patterns on the ground using rice flour, 
            chalk powder, or other natural materials.
            
            #### Historical Timeline:
            
            **🏺 Ancient Period (3000+ years ago)**
            - Evidence found in Indus Valley Civilization
            - Mentioned in ancient Tamil literature
            - Originally used rice flour to feed ants and small creatures
            
            **📚 Classical Period (500 BCE - 500 CE)**
            - Documented in Sangam literature
            - Became part of daily spiritual practice
            - Integration with Hindu religious customs
            
            **🏰 Medieval Period (500 - 1500 CE)**
            - Patronage by Tamil kingdoms
            - Development of complex geometric patterns
            - Spread to other South Indian regions
            
            **🌅 Modern Era (1500 CE - Present)**
            - Colonial documentation and study
            - Revival movements in 20th century
            - Digital preservation and global recognition
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Quick Facts
            """)
            
            # Create a simple info box
            st.info("""
            **Age:** 5000+ years
            
            **Origin:** Tamil Nadu
            
            **Materials:** Rice flour, chalk
            
            **Purpose:** Spiritual, decorative
            
            **Timing:** Daily, dawn hours
            
            **Practitioners:** Primarily women
            """)
            
            # Timeline visualization
            years = [3000, 500, 0, 500, 1500, 2000]
            events = ['Indus Valley', 'Sangam Era', 'Classical', 'Medieval', 'Colonial', 'Digital']
            
            fig = go.Figure(data=go.Scatter(
                x=years,
                y=[1]*len(years),
                mode='markers+text',
                text=events,
                textposition="top center",
                marker=dict(size=15, color='orange')
            ))
            
            fig.update_layout(
                title="Kolam Through History",
                xaxis_title="Years (BCE/CE)",
                yaxis=dict(visible=False),
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🕉️ Cultural & Spiritual Meaning")
        
        st.markdown("""
        ### Deep Cultural Significance
        
        Kolam is much more than decorative art - it's a profound spiritual and cultural practice 
        that reflects the Tamil worldview and philosophy.
        """)
        
        # Cultural aspects in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🌅 Daily Spiritual Practice
            
            **Morning Ritual:**
            - Created at dawn before sunrise
            - Purifies the entrance of homes
            - Invites positive energy and prosperity
            - Marks the beginning of each day
            
            **Spiritual Beliefs:**
            - Connects earthly and divine realms
            - Represents cosmic energy flow
            - Symbolizes the eternal cycle of creation
            - Offers protection from negative forces
            
            #### 👩 Women's Tradition
            
            **Cultural Role:**
            - Primarily practiced by women
            - Passed down through generations
            - Expression of creativity and devotion
            - Community bonding activity
            """)
        
        with col2:
            st.markdown("""
            #### 🌾 Environmental Harmony
            
            **Natural Materials:**
            - Rice flour feeds ants and insects
            - Chalk powder is biodegradable
            - Promotes ecological balance
            - Demonstrates respect for nature
            
            **Seasonal Patterns:**
            - Different designs for festivals
            - Harvest celebrations reflected in motifs
            - Weather-responsive practices
            - Agricultural cycle integration
            
            #### 🎭 Festivals & Celebrations
            
            **Special Occasions:**
            - Margazhi month (Dec-Jan) competitions
            - Pongal harvest festival elaborate designs
            - Wedding ceremonies grand patterns
            - Religious festivals themed motifs
            """)
        
        # Interactive cultural element selector
        st.subheader("🔍 Explore Cultural Elements")
        
        cultural_element = st.selectbox(
            "Select Cultural Aspect to Learn More:",
            [
                "Spiritual Symbolism",
                "Festival Traditions",
                "Regional Practices", 
                "Modern Adaptations"
            ]
        )
        
        if cultural_element == "Spiritual Symbolism":
            st.markdown("""
            ### 🕉️ Spiritual Symbolism in Kolam
            
            Every element in Kolam design carries deep spiritual meaning:
            
            - **Dots (Pulli):** Represent the universe's fundamental particles
            - **Lines:** Symbolize life's journey and connections
            - **Closed Loops:** Indicate the cyclical nature of existence
            - **Symmetry:** Reflects cosmic harmony and balance
            - **Center Point:** Represents the divine source of creation
            """)
        
        elif cultural_element == "Festival Traditions":
            st.markdown("""
            ### 🎉 Festival Kolam Traditions
            
            Special occasions call for elaborate Kolam designs:
            
            **Pongal Festival:**
            - Large, colorful patterns
            - Harvest motifs (sugarcane, rice, sun)
            - Community participation
            
            **Margazhi Month:**
            - Daily competitions between households
            - Increasingly complex patterns
            - Traditional versus modern judging criteria
            
            **Weddings:**
            - Grand entrance decorations
            - Auspicious symbols incorporated
            - Professional artists often employed
            """)
    
    with tab3:
        st.subheader("🗺️ Regional Variations")
        
        st.markdown("Kolam art has evolved differently across various regions of South India:")
        
        # Regional variations map (conceptual)
        regions_data = {
            'Region': ['Tamil Nadu', 'Karnataka', 'Andhra Pradesh', 'Kerala', 'Sri Lanka'],
            'Local Name': ['Kolam', 'Rangoli', 'Muggulu', 'Kalam', 'Kolam'],
            'Distinctive Features': [
                'Geometric precision, dot-based',
                'Colorful, floral themes',
                'Large-scale, festive designs',
                'Ritualistic, temple-related',
                'Preserved traditional methods'
            ],
            'Primary Materials': [
                'Rice flour, chalk',
                'Colored powders',
                'Colored rice, flowers',
                'Rice paste, flowers',
                'Rice flour, natural colors'
            ]
        }
        
        regions_df = pd.DataFrame(regions_data)
        st.dataframe(regions_df, use_container_width=True)
        
        # Interactive regional selector
        selected_region = st.selectbox("Explore Regional Style:", regions_data['Region'])
        
        region_index = regions_data['Region'].index(selected_region)
        
        st.markdown(f"""
        ### {selected_region} Style Details
        
        **Local Name:** {regions_data['Local Name'][region_index]}
        
        **Distinctive Features:** {regions_data['Distinctive Features'][region_index]}
        
        **Primary Materials:** {regions_data['Primary Materials'][region_index]}
        """)
    
    with tab4:
        st.subheader("🌐 Modern Evolution & Global Recognition")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### 🏆 Recognition & Preservation
            
            **UNESCO Recognition:**
            - Intangible Cultural Heritage consideration
            - Documentation projects worldwide
            - Academic research initiatives
            
            **Digital Age Adaptation:**
            - Mobile apps for learning
            - Online communities and competitions
            - AI-powered pattern generation
            
            **Global Spread:**
            - International cultural festivals
            - Art exhibitions worldwide
            - Educational workshops abroad
            """)
        
        with col2:
            st.markdown("""
            ### 📱 Contemporary Innovations
            
            **Modern Materials:**
            - Washable chalk alternatives
            - LED-enhanced patterns
            - 3D projection mapping
            
            **New Platforms:**
            - Social media challenges
            - Digital art installations
            - Interactive museum exhibits
            
            **Fusion Art:**
            - Contemporary art integration
            - Architecture-inspired designs
            - Cross-cultural adaptations
            """)

elif section == "🔢 Mathematics in Kolam":
    st.header("🔢 The Mathematical Beauty of Kolam")
    
    st.markdown("""
    Kolam patterns are treasure troves of mathematical concepts, demonstrating sophisticated 
    understanding of geometry, symmetry, and mathematical principles.
    """)
    
    # Mathematics tabs
    math_tab1, math_tab2, math_tab3, math_tab4 = st.tabs([
        "Geometric Foundations", 
        "Symmetry Principles", 
        "Fractal Properties", 
        "Topology & Graph Theory"
    ])
    
    with math_tab1:
        st.subheader("📐 Geometric Foundations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Grid-Based Geometry
            
            Kolam patterns are built on mathematical grids that provide the foundation for complex designs.
            
            #### Grid Systems:
            
            **1. Dot Grid (Pulli):**
            - Regular array of equally spaced dots
            - Typically square or rectangular arrangement  
            - Grid sizes: 3×3, 5×5, 7×7, up to 15×15 or larger
            - Dots serve as connection vertices
            
            **2. Mathematical Properties:**
            - **Spacing:** Uniform distance between adjacent dots
            - **Angles:** 90° angles in square grids
            - **Coordinates:** Each dot has (x,y) position
            - **Symmetry axes:** Multiple lines of symmetry
            """)
            
            # Interactive grid demonstration
            st.subheader("🎯 Interactive Grid Explorer")
            
            grid_demo_size = st.slider("Grid Size", 3, 9, 5, step=2)
            
            # Create visualization of dot grid
            fig = go.Figure()
            
            # Generate grid points
            x_coords = []
            y_coords = []
            for i in range(grid_demo_size):
                for j in range(grid_demo_size):
                    x_coords.append(i)
                    y_coords.append(j)
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Grid Dots'
            ))
            
            fig.update_layout(
                title=f"{grid_demo_size}×{grid_demo_size} Dot Grid",
                xaxis=dict(showgrid=True, gridwidth=1),
                yaxis=dict(showgrid=True, gridwidth=1),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            ### 📊 Grid Statistics
            """)
            
            total_dots = grid_demo_size ** 2
            max_connections = total_dots * (total_dots - 1) // 2
            perimeter_dots = 4 * (grid_demo_size - 1) if grid_demo_size > 1 else 1
            inner_dots = max(0, (grid_demo_size - 2) ** 2)
            
            st.metric("Total Dots", total_dots)
            st.metric("Perimeter Dots", perimeter_dots)
            st.metric("Inner Dots", inner_dots)
            st.metric("Max Connections", max_connections)
            
            st.markdown("""
            ### 🔍 Geometric Concepts
            
            **Distance Calculations:**
            - Euclidean distance between dots
            - Manhattan distance alternatives
            - Diagonal vs. orthogonal movements
            
            **Angle Relationships:**
            - 0°, 45°, 90° primary angles
            - Circular arc approximations
            - Tangent calculations for curves
            """)
    
    with math_tab2:
        st.subheader("🔄 Symmetry Principles")
        
        st.markdown("""
        ### Types of Symmetry in Kolam
        
        Kolam patterns demonstrate various mathematical symmetries, each with unique properties and beauty.
        """)
        
        # Symmetry type selector
        symmetry_type = st.selectbox(
            "Explore Symmetry Type:",
            ["Bilateral (Reflection)", "Rotational", "Translational", "Glide Reflection"]
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if symmetry_type == "Bilateral (Reflection)":
                st.markdown("""
                #### 🪞 Bilateral Symmetry
                
                **Definition:** Pattern looks identical when reflected across an axis
                
                **Mathematical Properties:**
                - Line of symmetry divides pattern into mirror halves
                - Points (x,y) map to (-x,y) for vertical axis
                - Preserves distances and angles
                - Most common in traditional Kolam
                
                **Applications:**
                - Human and animal motifs
                - Floral designs
                - Architectural elements
                """)
                
            elif symmetry_type == "Rotational":
                st.markdown("""
                #### 🌀 Rotational Symmetry
                
                **Definition:** Pattern looks identical when rotated by specific angles
                
                **Mathematical Properties:**
                - Rotation angle = 360°/n (where n = order of symmetry)
                - Common orders: 2, 3, 4, 6, 8
                - Center point remains fixed
                - Preserves shape and size
                
                **Applications:**
                - Star patterns
                - Mandala-style designs
                - Geometric motifs
                """)
                
            elif symmetry_type == "Translational":
                st.markdown("""
                #### ➡️ Translational Symmetry
                
                **Definition:** Pattern repeats at regular intervals
                
                **Mathematical Properties:**
                - Translation vector defines repeat distance
                - Pattern extends infinitely in principle
                - Preserves orientation and scale
                - Creates rhythmic visual flow
                
                **Applications:**
                - Border patterns
                - Wallpaper-like designs
                - Infinite grid extensions
                """)
        
        with col2:
            # Create symmetry visualization
            fig = go.Figure()
            
            if symmetry_type == "Bilateral (Reflection)":
                # Create a simple bilateral pattern
                x_points = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0]
                y_points = [0, 1, 0, 2, 3, 2, 4, 2, 3, 2, 0, 1, 0]
                
                fig.add_trace(go.Scatter(x=x_points, y=y_points, mode='lines+markers', name='Pattern'))
                fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=5, 
                             line=dict(color="red", dash="dash"), name="Symmetry Axis")
                
            elif symmetry_type == "Rotational":
                # Create rotational pattern
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                for angle in angles:
                    x_rot = np.cos(angle) * 2
                    y_rot = np.sin(angle) * 2
                    fig.add_trace(go.Scatter(x=[0, x_rot], y=[0, y_rot], mode='lines+markers', 
                                           showlegend=False))
            
            fig.update_layout(title=f"{symmetry_type} Example", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Symmetry analysis tool
        st.subheader("🔧 Symmetry Analysis Tool")
        
        st.markdown("Upload a pattern to analyze its symmetries:")
        
        symmetry_test = st.selectbox(
            "Test Symmetry Detection:",
            ["Demo Pattern 1", "Demo Pattern 2", "Demo Pattern 3"]
        )
        
        # Mock symmetry scores (in real app, these would be calculated)
        bilateral_score = np.random.random()
        rotational_score = np.random.random()
        translational_score = np.random.random()
        
        symm_col1, symm_col2, symm_col3 = st.columns(3)
        
        with symm_col1:
            st.metric("Bilateral", f"{bilateral_score:.3f}")
            st.progress(bilateral_score)
        
        with symm_col2:
            st.metric("Rotational", f"{rotational_score:.3f}")
            st.progress(rotational_score)
        
        with symm_col3:
            st.metric("Translational", f"{translational_score:.3f}")
            st.progress(translational_score)
    
    with math_tab3:
        st.subheader("🌿 Fractal Properties")
        
        st.markdown("""
        ### Self-Similarity in Kolam Patterns
        
        Many Kolam designs exhibit fractal properties, where patterns repeat at different scales,
        creating infinite complexity from simple rules.
        """)
        
        # Fractal concepts
        fractal_col1, fractal_col2 = st.columns([1, 1])
        
        with fractal_col1:
            st.markdown("""
            #### 🔍 Fractal Characteristics
            
            **Self-Similarity:**
            - Small parts resemble the whole pattern
            - Recursive geometric construction
            - Scale-invariant properties
            
            **Mathematical Measures:**
            - **Fractal Dimension:** Non-integer dimension value
            - **Hausdorff Dimension:** Theoretical measure
            - **Box-Counting Method:** Practical calculation
            
            **Examples in Kolam:**
            - Nested geometric shapes
            - Recursive spiral patterns
            - Tree-like branching structures
            """)
            
            # Fractal dimension calculator
            st.subheader("📐 Fractal Dimension Calculator")
            
            pattern_complexity = st.slider("Pattern Complexity", 1, 10, 5)
            scaling_factor = st.slider("Scaling Factor", 2, 5, 3)
            
            # Mock fractal dimension calculation
            fractal_dim = 1 + (pattern_complexity / 10) * np.log(scaling_factor) / np.log(2)
            
            st.metric("Estimated Fractal Dimension", f"{fractal_dim:.3f}")
            
            if fractal_dim < 1.5:
                st.info("Simple geometric pattern")
            elif fractal_dim < 2.0:
                st.warning("Moderate fractal complexity")
            else:
                st.success("High fractal complexity")
        
        with fractal_col2:
            # Generate fractal-like pattern visualization
            st.subheader("🌿 Fractal Pattern Generator")
            
            iterations = st.slider("Iterations", 1, 6, 3)
            
            # Create simple fractal visualization
            fig = go.Figure()
            
            def add_fractal_branch(x, y, length, angle, iteration, max_iter):
                if iteration > max_iter:
                    return
                
                x2 = x + length * np.cos(angle)
                y2 = y + length * np.sin(angle)
                
                fig.add_trace(go.Scatter(
                    x=[x, x2], y=[y, y2], 
                    mode='lines', 
                    line=dict(color=f'rgb({int(255*iteration/max_iter)}, 100, {int(255*(max_iter-iteration)/max_iter)})'),
                    showlegend=False
                ))
                
                new_length = length * 0.7
                add_fractal_branch(x2, y2, new_length, angle + np.pi/4, iteration + 1, max_iter)
                add_fractal_branch(x2, y2, new_length, angle - np.pi/4, iteration + 1, max_iter)
            
            # Generate fractal tree
            add_fractal_branch(0, 0, 1, np.pi/2, 1, iterations)
            
            fig.update_layout(
                title="Fractal Tree Pattern",
                height=400,
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with math_tab4:
        st.subheader("🕸️ Topology & Graph Theory")
        
        st.markdown("""
        ### Graph Theory Applications in Kolam
        
        Kolam patterns can be analyzed using graph theory, where dots become vertices 
        and connecting lines become edges.
        """)
        
        # Graph theory concepts
        graph_col1, graph_col2 = st.columns([1, 1])
        
        with graph_col1:
            st.markdown("""
            #### 🔗 Graph Theory Basics
            
            **Vertices (Dots):**
            - Grid intersection points
            - Connection nodes in the pattern
            - Degree = number of connected edges
            
            **Edges (Lines):**
            - Connections between vertices
            - Can be weighted (line thickness)
            - May have direction (flow patterns)
            
            **Graph Properties:**
            - **Connectivity:** All vertices reachable
            - **Planarity:** Can be drawn without edge crossings
            - **Cycles:** Closed loops in the pattern
            - **Eulerian Paths:** Visit each edge exactly once
            """)
            
            # Graph analysis parameters
            st.subheader("📊 Graph Analysis")
            
            vertex_count = st.number_input("Number of Vertices", 4, 25, 9)
            edge_density = st.slider("Edge Density", 0.1, 1.0, 0.4)
            
            edge_count = int(vertex_count * (vertex_count - 1) * edge_density / 2)
            
            st.metric("Vertices", vertex_count)
            st.metric("Edges", edge_count)
            st.metric("Graph Density", f"{edge_density:.1%}")
            
            # Graph properties
            avg_degree = 2 * edge_count / vertex_count
            st.metric("Average Degree", f"{avg_degree:.1f}")
        
        with graph_col2:
            st.subheader("🌐 Graph Visualization")
            
            # Generate random graph for demonstration
            np.random.seed(42)
            
            # Create vertices
            vertices_x = np.random.uniform(-1, 1, vertex_count)
            vertices_y = np.random.uniform(-1, 1, vertex_count)
            
            fig = go.Figure()
            
            # Add edges
            edges_added = 0
            for i in range(vertex_count):
                for j in range(i + 1, vertex_count):
                    if np.random.random() < edge_density and edges_added < edge_count:
                        fig.add_trace(go.Scatter(
                            x=[vertices_x[i], vertices_x[j]], 
                            y=[vertices_y[i], vertices_y[j]],
                            mode='lines',
                            line=dict(color='lightblue', width=1),
                            showlegend=False
                        ))
                        edges_added += 1
            
            # Add vertices
            fig.add_trace(go.Scatter(
                x=vertices_x,
                y=vertices_y,
                mode='markers',
                marker=dict(size=10, color='red'),
                showlegend=False
            ))
            
            fig.update_layout(
                title="Graph Representation",
                height=400,
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Eulerian path analysis
        st.subheader("🛤️ Eulerian Path Analysis")
        
        st.markdown("""
        **Eulerian Path:** A path that visits every edge exactly once.
        
        Traditional Kolam drawing follows this principle - the pattern is drawn in one continuous 
        line without lifting the drawing tool.
        """)
        
        eulerian_col1, eulerian_col2 = st.columns(2)
        
        with eulerian_col1:
            st.markdown("""
            **Conditions for Eulerian Path:**
            - Graph must be connected
            - Exactly 0 or 2 vertices of odd degree
            - If 0 odd vertices: Eulerian circuit exists
            - If 2 odd vertices: Eulerian path exists
            """)
        
        with eulerian_col2:
            # Check Eulerian path possibility
            odd_degree_vertices = sum(1 for _ in range(vertex_count) if np.random.randint(1, 5) % 2 == 1)
            
            if odd_degree_vertices == 0:
                st.success("✅ Eulerian circuit possible!")
                st.info("Can be drawn starting and ending at same point")
            elif odd_degree_vertices == 2:
                st.success("✅ Eulerian path possible!")
                st.info("Can be drawn in one continuous line")
            else:
                st.warning("❌ No Eulerian path possible")
                st.info("Would require lifting the drawing tool")

elif section == "🎨 Types of Kolam":
    st.header("🎨 Different Types of Kolam Patterns")
    
    # Pattern type selector
    pattern_type = st.selectbox(
        "Explore Pattern Type:",
        [
            "Pulli Kolam (Dot Patterns)",
            "Sikku Kolam (Line Patterns)", 
            "Kambi Kolam (Rope Patterns)",
            "Neli Kolam (Paddy Patterns)",
            "Poo Kolam (Flower Patterns)"
        ]
    )
    
    if pattern_type == "Pulli Kolam (Dot Patterns)":
        st.subheader("⚫ Pulli Kolam - The Foundation of All Patterns")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### What is Pulli Kolam?
            
            **Pulli** means "dot" in Tamil. These patterns start with a grid of dots that serve as 
            the foundation for creating intricate designs. The dots act as anchor points that 
            guide the formation of curves, loops, and geometric shapes.
            
            #### Key Characteristics:
            
            **🎯 Dot Grid Foundation:**
            - Regular arrangement of dots in rows and columns
            - Dots are evenly spaced for symmetry
            - Grid size determines pattern complexity
            - Common sizes: 3×3, 5×5, 7×7, 9×9, 11×11
            
            **🔄 Connection Rules:**
            - Lines must connect dots smoothly
            - No sharp angles at dot connections
            - Curves should flow naturally between dots
            - All dots should ideally be incorporated
            
            **📐 Geometric Principles:**
            - Maintains bilateral or rotational symmetry
            - Creates enclosed spaces and open pathways
            - Follows mathematical proportions
            - Demonstrates grid-based geometry
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Pulli Pattern Statistics
            """)
            
            # Interactive dot grid generator
            pulli_size = st.slider("Dot Grid Size", 3, 11, 5, step=2)
            dot_spacing = st.slider("Dot Spacing", 1.0, 3.0, 2.0, step=0.5)
            
            total_dots = pulli_size ** 2
            perimeter_dots = 4 * (pulli_size - 1) if pulli_size > 1 else 1
            
            st.metric("Total Dots", total_dots)
            st.metric("Perimeter Dots", perimeter_dots)
            st.metric("Inner Dots", max(0, (pulli_size - 2) ** 2))
            
            # Generate dot pattern visualization
            fig = go.Figure()
            
            for i in range(pulli_size):
                for j in range(pulli_size):
                    fig.add_trace(go.Scatter(
                        x=[i * dot_spacing],
                        y=[j * dot_spacing],
                        mode='markers',
                        marker=dict(size=12, color='red'),
                        showlegend=False
                    ))
            
            fig.update_layout(
                title=f"{pulli_size}×{pulli_size} Pulli Grid",
                height=300,
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Pulli pattern examples and tutorial
        st.subheader("📚 Learn to Draw Pulli Kolam")
        
        tutorial_step = st.selectbox(
            "Tutorial Step:",
            [
                "Step 1: Create the Dot Grid",
                "Step 2: Connect Corner Dots", 
                "Step 3: Add Curved Lines",
                "Step 4: Create Enclosed Shapes",
                "Step 5: Final Decorative Elements"
            ]
        )
        
        # Tutorial content based on selected step
        if tutorial_step == "Step 1: Create the Dot Grid":
            st.markdown("""
            #### 🎯 Creating the Perfect Dot Grid
            
            **Materials Needed:**
            - Rice flour or chalk powder
            - Steady hand and patience
            - Clean, flat surface
            
            **Technique:**
            1. Start with a small amount of flour between thumb and forefinger
            2. Create dots by gently touching the ground
            3. Maintain equal spacing between dots
            4. Work systematically row by row
            5. Ensure dots are clearly visible and uniform
            
            **Tips for Beginners:**
            - Start with 3×3 or 5×5 grids
            - Practice dot consistency before attempting patterns
            - Use a light touch to avoid smudging
            """)
        
        elif tutorial_step == "Step 2: Connect Corner Dots":
            st.markdown("""
            #### 🔗 Connecting the Dots
            
            **Basic Connection Rules:**
            1. Start from any corner or edge dot
            2. Draw smooth curves to adjacent dots
            3. Avoid sharp angles at connection points
            4. Maintain symmetry while drawing
            
            **Common Connection Patterns:**
            - **Straight Lines:** Direct connections between dots
            - **Curved Arcs:** Smooth curves that loop around dots
            - **S-Curves:** Serpentine connections creating flow
            - **Loops:** Circular or oval shapes around single dots
            """)
        
        # Interactive pattern builder
        st.subheader("🎨 Interactive Pulli Pattern Builder")
        
        st.markdown("Create your own Pulli Kolam step by step:")
        
        builder_col1, builder_col2 = st.columns([1, 1])
        
        with builder_col1:
            pattern_complexity = st.selectbox(
                "Pattern Complexity:",
                ["Beginner", "Intermediate", "Advanced"]
            )
            
            include_curves = st.checkbox("Include Curved Lines", value=True)
            include_loops = st.checkbox("Include Loops", value=True)
            symmetric_pattern = st.checkbox("Force Symmetry", value=True)
        
        with builder_col2:
            if st.button("Generate Pulli Pattern", type="primary"):
                st.success("✅ Pattern generated! (In full app, this would show actual pattern)")
                
                # Mock pattern generation result
                st.info(f"""
                **Generated Pattern Properties:**
                - Grid Size: {pulli_size}×{pulli_size}
                - Complexity: {pattern_complexity}
                - Symmetry: {'Yes' if symmetric_pattern else 'No'}
                - Elements: {', '.join([x for x in ['Curves', 'Loops', 'Lines'] if True])}
                """)

    elif pattern_type == "Sikku Kolam (Line Patterns)":
        st.subheader("🌀 Sikku Kolam - Interlaced Beauty")
        
        st.markdown("""
        ### The Art of Interlacing
        
        **Sikku** means "to entangle" or "to interweave." These patterns feature complex 
        interlaced lines that weave over and under each other, creating stunning visual 
        effects and demonstrating advanced geometric understanding.
        """)
        
        # Sikku characteristics
        sikku_col1, sikku_col2 = st.columns([1, 1])
        
        with sikku_col1:
            st.markdown("""
            #### 🕷️ Key Features:
            
            **Interlacing Principle:**
            - Lines weave alternately over and under
            - Creates 3D visual effects on 2D surface
            - Demonstrates topological complexity
            - Requires careful planning and execution
            
            **Mathematical Concepts:**
            - **Topology:** Study of spatial properties
            - **Knot Theory:** Mathematical analysis of loops
            - **Graph Theory:** Network of intersections
            - **Alternating Patterns:** Regular over/under sequence
            
            **Traditional Significance:**
            - Represents life's interconnectedness
            - Symbolizes the weaving of destiny
            - Shows complexity emerging from simplicity
            - Demonstrates mathematical beauty
            """)
        
        with sikku_col2:
            st.markdown("""
            #### 🎯 Drawing Techniques:
            
            **Planning Phase:**
            1. Sketch the basic framework
            2. Identify crossing points
            3. Plan the over/under sequence
            4. Mark the path direction
            
            **Execution Phase:**
            1. Draw the base lines lightly
            2. Add interlacing details
            3. Ensure consistent alternation
            4. Maintain line continuity
            
            **Common Mistakes:**
            - Inconsistent over/under pattern
            - Broken continuity at crossings
            - Uneven line spacing
            - Lost track of path direction
            """)
        
        # Interactive interlacing demo
        st.subheader("🔧 Interactive Interlacing Demo")
        
        interlace_demo = st.selectbox(
            "Choose Interlacing Pattern:",
            ["Simple Weave", "Celtic Knot", "Infinity Loop", "Star Interlace"]
        )
        
        if interlace_demo == "Simple Weave":
            st.markdown("""
            #### Simple Weave Pattern
            
            This basic interlacing creates a simple over-under pattern:
            
            ```
            ╭─────╮
            │  ╱  │  ← Line A goes over Line B
            │╱    │
            ╰─────╯
            ```
            
            Practice this basic weave before moving to complex patterns.
            """)
