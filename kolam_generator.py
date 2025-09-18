import numpy as np
import math
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
import logging
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KolamGenerator:
    """
    Algorithmic Kolam pattern generator using mathematical principles
    """
    
    def __init__(self):
        self.logger = logger
        self.presets = self._initialize_presets()
        logger.info("Kolam generator initialized")
    
    def _initialize_presets(self):
        """Initialize pattern presets"""
        return {
            'floral': {
                'curve_style': 'Flowing',
                'complexity': 'Medium',
                'symmetry_type': 'Radial',
                'theme': 'Floral',
                'decorations': True
            },
            'geometric': {
                'curve_style': 'Angular',
                'complexity': 'Simple',
                'symmetry_type': 'Bilateral',
                'theme': 'Geometric',
                'decorations': False
            },
            'traditional': {
                'curve_style': 'Smooth',
                'complexity': 'Medium',
                'symmetry_type': 'Bilateral',
                'theme': 'Traditional',
                'decorations': True
            },
            'modern': {
                'curve_style': 'Mixed',
                'complexity': 'Complex',
                'symmetry_type': 'Mixed',
                'theme': 'Modern',
                'decorations': True
            }
        }
    
    def generate_pattern(self, params):
        """
        Generate Kolam pattern based on parameters
        
        Args:
            params: Dictionary with generation parameters
            
        Returns:
            Dictionary with generated pattern data
        """
        try:
            start_time = time.time()
            
            # Extract parameters
            grid_size = params.get('grid_size', 7)
            dot_density = params.get('dot_density', 0.7)
            symmetry_type = params.get('symmetry_type', 'Bilateral')
            complexity = params.get('complexity', 'Medium')
            curve_style = params.get('curve_style', 'Smooth')
            theme = params.get('theme', 'Traditional')
            line_thickness = params.get('line_thickness', 3)
            smoothness = params.get('smoothness', 1.0)
            decorations = params.get('decorations', True)
            
            # Canvas setup
            canvas_size = 600
            margin = 50
            working_area = canvas_size - 2 * margin
            
            # Create image
            image = Image.new('RGB', (canvas_size, canvas_size), 'black')
            draw = ImageDraw.Draw(image)
            
            # Generate dot grid
            dots = self._generate_dot_grid(grid_size, working_area, margin, dot_density)
            
            # Generate pattern based on theme and parameters
            if theme == 'Geometric':
                pattern_data = self._generate_geometric_pattern(draw, dots, params)
            elif theme == 'Floral':
                pattern_data = self._generate_floral_pattern(draw, dots, params)
            elif theme == 'Traditional':
                pattern_data = self._generate_traditional_pattern(draw, dots, params)
            elif theme == 'Modern':
                pattern_data = self._generate_modern_pattern(draw, dots, params)
            else:
                pattern_data = self._generate_basic_pattern(draw, dots, params)
            
            # Apply symmetry
            image = self._apply_symmetry(image, symmetry_type, params)
            
            # Add decorative elements if requested
            if decorations:
                image = self._add_decorative_elements(image, params)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Prepare result
            result = {
                'image': image,
                'dot_count': len(dots),
                'line_count': pattern_data.get('line_count', 0),
                'generation_time': generation_time,
                'geometric_properties': self._calculate_geometric_properties(image),
                'mathematical_metrics': self._calculate_mathematical_metrics(image),
                'algorithm_details': pattern_data.get('algorithm_details', [])
            }
            
            logger.info(f"Pattern generated successfully in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating pattern: {str(e)}")
            return None
    
    def _generate_dot_grid(self, grid_size, working_area, margin, density):
        """
        Generate dot grid for pattern foundation
        
        Args:
            grid_size: Size of the grid (NxN)
            working_area: Available drawing area
            margin: Margin from edges
            density: Dot density (0-1)
            
        Returns:
            List of dot coordinates
        """
        try:
            dots = []
            spacing = working_area / (grid_size - 1) if grid_size > 1 else 0
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if random.random() < density:
                        x = margin + i * spacing
                        y = margin + j * spacing
                        dots.append((x, y))
            
            return dots
            
        except Exception as e:
            logger.error(f"Error generating dot grid: {str(e)}")
            return []
    
    def _generate_geometric_pattern(self, draw, dots, params):
        """
        Generate geometric Kolam pattern
        
        Args:
            draw: ImageDraw object
            dots: List of dot coordinates
            params: Generation parameters
            
        Returns:
            Pattern generation data
        """
        try:
            line_count = 0
            algorithm_details = ["Geometric pattern generation"]
            
            line_thickness = params.get('line_thickness', 3)
            complexity = params.get('complexity', 'Medium')
            
            if len(dots) < 4:
                return {'line_count': 0, 'algorithm_details': algorithm_details}
            
            # Sort dots for systematic connection
            dots_sorted = sorted(dots, key=lambda p: (p[0], p[1]))
            
            # Draw connecting lines between adjacent dots
            complexity_factor = {'Simple': 0.3, 'Medium': 0.6, 'Complex': 0.9}.get(complexity, 0.6)
            
            for i, dot1 in enumerate(dots_sorted):
                # Connect to nearby dots based on complexity
                nearby_dots = self._find_nearby_dots(dot1, dots_sorted, complexity_factor)
                
                for dot2 in nearby_dots:
                    if self._should_connect_dots(dot1, dot2, complexity_factor):
                        draw.line([dot1, dot2], fill='white', width=line_thickness)
                        line_count += 1
            
            # Add geometric shapes
            if complexity in ['Medium', 'Complex']:
                line_count += self._add_geometric_shapes(draw, dots, params)
                algorithm_details.append("Added geometric shapes")
            
            # Draw dots
            for dot in dots:
                draw.ellipse([dot[0]-2, dot[1]-2, dot[0]+2, dot[1]+2], fill='white')
            
            algorithm_details.append(f"Connected {line_count} lines")
            
            return {
                'line_count': line_count,
                'algorithm_details': algorithm_details
            }
            
        except Exception as e:
            logger.error(f"Error generating geometric pattern: {str(e)}")
            return {'line_count': 0, 'algorithm_details': [f"Error: {str(e)}"]}
    
    def _generate_floral_pattern(self, draw, dots, params):
        """
        Generate floral Kolam pattern
        """
        try:
            line_count = 0
            algorithm_details = ["Floral pattern generation"]
            
            line_thickness = params.get('line_thickness', 3)
            
            if len(dots) < 1:
                return {'line_count': 0, 'algorithm_details': algorithm_details}
            
            # Create flower-like patterns around dots
            for dot in dots:
                if random.random() < 0.7:  # 70% chance for each dot
                    petals = self._draw_flower_petals(draw, dot, params)
                    line_count += petals
            
            # Connect flowers with curved stems
            if len(dots) > 1:
                stems = self._draw_curved_connections(draw, dots, params)
                line_count += stems
                algorithm_details.append("Added curved stem connections")
            
            algorithm_details.append(f"Generated {len(dots)} floral elements")
            
            return {
                'line_count': line_count,
                'algorithm_details': algorithm_details
            }
            
        except Exception as e:
            logger.error(f"Error generating floral pattern: {str(e)}")
            return {'line_count': 0, 'algorithm_details': [f"Error: {str(e)}"]}
    
    def _generate_traditional_pattern(self, draw, dots, params):
        """
        Generate traditional Kolam pattern
        """
        try:
            line_count = 0
            algorithm_details = ["Traditional Kolam generation"]
            
            if len(dots) < 4:
                return {'line_count': 0, 'algorithm_details': algorithm_details}
            
            # Traditional Kolam follows specific rules
            # 1. Create continuous loops around dots
            line_count += self._create_traditional_loops(draw, dots, params)
            algorithm_details.append("Created traditional continuous loops")
            
            # 2. Add traditional motifs
            if params.get('decorations', True):
                motifs = self._add_traditional_motifs(draw, dots, params)
                line_count += motifs
                algorithm_details.append("Added traditional motifs")
            
            # 3. Ensure symmetry
            algorithm_details.append("Applied traditional symmetry rules")
            
            return {
                'line_count': line_count,
                'algorithm_details': algorithm_details
            }
            
        except Exception as e:
            logger.error(f"Error generating traditional pattern: {str(e)}")
            return {'line_count': 0, 'algorithm_details': [f"Error: {str(e)}"]}
    
    def _generate_modern_pattern(self, draw, dots, params):
        """
        Generate modern interpretation of Kolam
        """
        try:
            line_count = 0
            algorithm_details = ["Modern Kolam generation"]
            
            # Modern patterns can break traditional rules
            # Mix different styles and add contemporary elements
            
            # Geometric base
            geom_lines = self._generate_geometric_pattern(draw, dots, params)['line_count']
            line_count += geom_lines
            
            # Add modern artistic elements
            if params.get('decorations', True):
                modern_elements = self._add_modern_elements(draw, dots, params)
                line_count += modern_elements
                algorithm_details.append("Added modern artistic elements")
            
            algorithm_details.append("Applied contemporary design principles")
            
            return {
                'line_count': line_count,
                'algorithm_details': algorithm_details
            }
            
        except Exception as e:
            logger.error(f"Error generating modern pattern: {str(e)}")
            return {'line_count': 0, 'algorithm_details': [f"Error: {str(e)}"]}
    
    def _generate_basic_pattern(self, draw, dots, params):
        """
        Generate basic fallback pattern
        """
        try:
            line_count = 0
            algorithm_details = ["Basic pattern generation"]
            
            line_thickness = params.get('line_thickness', 3)
            
            # Simple connections between adjacent dots
            for i in range(len(dots) - 1):
                draw.line([dots[i], dots[i+1]], fill='white', width=line_thickness)
                line_count += 1
            
            # Draw dots
            for dot in dots:
                draw.ellipse([dot[0]-3, dot[1]-3, dot[0]+3, dot[1]+3], fill='white')
            
            return {
                'line_count': line_count,
                'algorithm_details': algorithm_details
            }
            
        except Exception as e:
            logger.error(f"Error generating basic pattern: {str(e)}")
            return {'line_count': 0, 'algorithm_details': [f"Error: {str(e)}"]}
    
    def _find_nearby_dots(self, center_dot, all_dots, radius_factor):
        """Find dots within a certain radius"""
        try:
            max_distance = 100 * radius_factor
            nearby = []
            
            for dot in all_dots:
                if dot != center_dot:
                    distance = math.sqrt((dot[0] - center_dot[0])**2 + (dot[1] - center_dot[1])**2)
                    if distance <= max_distance:
                        nearby.append(dot)
            
            return nearby
            
        except Exception as e:
            logger.error(f"Error finding nearby dots: {str(e)}")
            return []
    
    def _should_connect_dots(self, dot1, dot2, complexity_factor):
        """Determine if two dots should be connected"""
        try:
            # Higher complexity = more connections
            connection_probability = complexity_factor * 0.8
            return random.random() < connection_probability
        except Exception as e:
            return False
    
    def _add_geometric_shapes(self, draw, dots, params):
        """Add geometric shapes to the pattern"""
        try:
            line_count = 0
            line_thickness = params.get('line_thickness', 3)
            
            if len(dots) >= 3:
                # Add some triangles
                for _ in range(min(3, len(dots) // 3)):
                    triangle_dots = random.sample(dots, 3)
                    draw.polygon(triangle_dots, outline='white', width=line_thickness)
                    line_count += 3
            
            if len(dots) >= 4:
                # Add some rectangles
                for _ in range(min(2, len(dots) // 6)):
                    rect_dots = random.sample(dots, 4)
                    # Sort to form proper rectangle
                    rect_dots.sort(key=lambda p: (p[0], p[1]))
                    if len(rect_dots) >= 4:
                        draw.polygon(rect_dots[:4], outline='white', width=line_thickness)
                        line_count += 4
            
            return line_count
            
        except Exception as e:
            logger.error(f"Error adding geometric shapes: {str(e)}")
            return 0
    
    def _draw_flower_petals(self, draw, center, params):
        """Draw flower petals around a center point"""
        try:
            line_count = 0
            line_thickness = params.get('line_thickness', 2)
            petal_count = random.randint(4, 8)
            petal_size = 20
            
            for i in range(petal_count):
                angle = (2 * math.pi * i) / petal_count
                petal_end = (
                    center[0] + petal_size * math.cos(angle),
                    center[1] + petal_size * math.sin(angle)
                )
                
                # Draw petal as an ellipse
                bbox = [
                    petal_end[0] - 5, petal_end[1] - 10,
                    petal_end[0] + 5, petal_end[1] + 10
                ]
                draw.ellipse(bbox, outline='white', width=line_thickness)
                line_count += 1
            
            return line_count
            
        except Exception as e:
            logger.error(f"Error drawing flower petals: {str(e)}")
            return 0
    
    def _draw_curved_connections(self, draw, dots, params):
        """Draw curved connections between dots"""
        try:
            line_count = 0
            line_thickness = params.get('line_thickness', 2)
            
            for i in range(0, len(dots) - 1, 2):
                if i + 1 < len(dots):
                    start = dots[i]
                    end = dots[i + 1]
                    
                    # Create curved line using multiple straight segments
                    mid_x = (start[0] + end[0]) / 2
                    mid_y = (start[1] + end[1]) / 2 - 30  # Curve upward
                    
                    # Draw curve as two line segments
                    draw.line([start, (mid_x, mid_y)], fill='white', width=line_thickness)
                    draw.line([(mid_x, mid_y), end], fill='white', width=line_thickness)
                    line_count += 2
            
            return line_count
            
        except Exception as e:
            logger.error(f"Error drawing curved connections: {str(e)}")
            return 0
    
    def _create_traditional_loops(self, draw, dots, params):
        """Create traditional continuous loops"""
        try:
            line_count = 0
            line_thickness = params.get('line_thickness', 3)
            
            # Group dots and create loops around them
            if len(dots) >= 4:
                # Create a simple loop connecting outer dots
                # Sort by distance from center
                center_x = sum(d[0] for d in dots) / len(dots)
                center_y = sum(d[1] for d in dots) / len(dots)
                
                sorted_dots = sorted(dots, key=lambda d: math.atan2(d[1] - center_y, d[0] - center_x))
                
                # Connect dots in circular fashion
                for i in range(len(sorted_dots)):
                    start = sorted_dots[i]
                    end = sorted_dots[(i + 1) % len(sorted_dots)]
                    draw.line([start, end], fill='white', width=line_thickness)
                    line_count += 1
            
            return line_count
            
        except Exception as e:
            logger.error(f"Error creating traditional loops: {str(e)}")
            return 0
    
    def _add_traditional_motifs(self, draw, dots, params):
        """Add traditional Kolam motifs"""
        try:
            line_count = 0
            line_thickness = params.get('line_thickness', 2)
            
            # Add small decorative elements around dots
            for dot in dots:
                if random.random() < 0.5:
                    # Add small cross pattern
                    size = 8
                    draw.line([
                        (dot[0] - size, dot[1]), 
                        (dot[0] + size, dot[1])
                    ], fill='white', width=line_thickness)
                    draw.line([
                        (dot[0], dot[1] - size), 
                        (dot[0], dot[1] + size)
                    ], fill='white', width=line_thickness)
                    line_count += 2
            
            return line_count
            
        except Exception as e:
            logger.error(f"Error adding traditional motifs: {str(e)}")
            return 0
    
    def _add_modern_elements(self, draw, dots, params):
        """Add modern artistic elements"""
        try:
            line_count = 0
            line_thickness = params.get('line_thickness', 2)
            
            # Add abstract geometric elements
            for dot in dots:
                if random.random() < 0.3:
                    # Add random geometric shape
                    shape_type = random.choice(['circle', 'triangle', 'line'])
                    
                    if shape_type == 'circle':
                        radius = random.randint(10, 25)
                        bbox = [
                            dot[0] - radius, dot[1] - radius,
                            dot[0] + radius, dot[1] + radius
                        ]
                        draw.ellipse(bbox, outline='white', width=line_thickness)
                        line_count += 1
                    
                    elif shape_type == 'triangle':
                        size = 15
                        triangle = [
                            (dot[0], dot[1] - size),
                            (dot[0] - size, dot[1] + size),
                            (dot[0] + size, dot[1] + size)
                        ]
                        draw.polygon(triangle, outline='white', width=line_thickness)
                        line_count += 3
            
            return line_count
            
        except Exception as e:
            logger.error(f"Error adding modern elements: {str(e)}")
            return 0
    
    def _apply_symmetry(self, image, symmetry_type, params):
        """Apply symmetry transformations to the pattern"""
        try:
            if symmetry_type == 'Bilateral':
                # Mirror horizontally
                width, height = image.size
                left_half = image.crop((0, 0, width//2, height))
                mirrored = left_half.transpose(Image.FLIP_LEFT_RIGHT)
                image.paste(mirrored, (width//2, 0))
            
            elif symmetry_type == 'Rotational':
                # Apply rotational symmetry (not implemented for simplicity)
                pass
            
            elif symmetry_type == 'Radial':
                # Apply radial symmetry (not implemented for simplicity)
                pass
            
            return image
            
        except Exception as e:
            logger.error(f"Error applying symmetry: {str(e)}")
            return image
    
    def _add_decorative_elements(self, image, params):
        """Add decorative elements to enhance the pattern"""
        try:
            # Apply some image enhancements
            if params.get('artistic_effects', False):
                # Add slight blur for softer look
                image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
                
                # Enhance contrast slightly
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)
            
            return image
            
        except Exception as e:
            logger.error(f"Error adding decorative elements: {str(e)}")
            return image
    
    def _calculate_geometric_properties(self, image):
        """Calculate geometric properties of the generated pattern"""
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            # Calculate basic properties
            total_pixels = gray_array.size
            white_pixels = np.sum(gray_array > 128)
            
            properties = {
                'area_coverage': float(white_pixels / total_pixels),
                'bilateral_symmetry': self._calculate_bilateral_symmetry(gray_array),
                'rotational_symmetry': 0.0,  # Placeholder
                'translational_symmetry': 0.0,  # Placeholder
                'complexity_index': self._calculate_complexity_index(gray_array)
            }
            
            return properties
            
        except Exception as e:
            logger.error(f"Error calculating geometric properties: {str(e)}")
            return {}
    
    def _calculate_bilateral_symmetry(self, gray_array):
        """Calculate bilateral symmetry score"""
        try:
            height, width = gray_array.shape
            left_half = gray_array[:, :width//2]
            right_half = gray_array[:, width//2:]
            right_flipped = np.fliplr(right_half)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_flipped = right_flipped[:, :min_width]
            
            # Calculate similarity
            diff = np.abs(left_half.astype(float) - right_flipped.astype(float))
            similarity = 1.0 - (np.mean(diff) / 255.0)
            
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"Error calculating bilateral symmetry: {str(e)}")
            return 0.0
    
    def _calculate_complexity_index(self, gray_array):
        """Calculate pattern complexity index"""
        try:
            # Use edge detection to measure complexity
            from scipy import ndimage
            
            # Calculate gradients
            grad_x = ndimage.sobel(gray_array, axis=1)
            grad_y = ndimage.sobel(gray_array, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize complexity
            max_gradient = np.max(gradient_magnitude)
            if max_gradient > 0:
                complexity = np.mean(gradient_magnitude) / max_gradient
            else:
                complexity = 0.0
            
            return float(complexity)
            
        except Exception as e:
            logger.error(f"Error calculating complexity index: {str(e)}")
            return 0.0
    
    def _calculate_mathematical_metrics(self, image):
        """Calculate mathematical metrics for the pattern"""
        try:
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            metrics = {
                'entropy': self._calculate_entropy(gray_array),
                'contrast': self._calculate_contrast(gray_array),
                'homogeneity': self._calculate_homogeneity(gray_array),
                'energy': self._calculate_energy(gray_array)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating mathematical metrics: {str(e)}")
            return {}
    
    def _calculate_entropy(self, gray_array):
        """Calculate image entropy"""
        try:
            hist, _ = np.histogram(gray_array, bins=256, range=(0, 256))
            hist = hist / np.sum(hist)  # Normalize
            hist = hist[hist > 0]  # Remove zeros
            entropy = -np.sum(hist * np.log2(hist))
            return float(entropy)
        except Exception as e:
            return 0.0
    
    def _calculate_contrast(self, gray_array):
        """Calculate image contrast"""
        try:
            return float(np.std(gray_array))
        except Exception as e:
            return 0.0
    
    def _calculate_homogeneity(self, gray_array):
        """Calculate image homogeneity"""
        try:
            # Simple homogeneity measure
            local_variance = ndimage.generic_filter(gray_array.astype(float), np.var, size=3)
            homogeneity = 1.0 / (1.0 + np.mean(local_variance))
            return float(homogeneity)
        except Exception as e:
            return 0.0
    
    def _calculate_energy(self, gray_array):
        """Calculate image energy"""
        try:
            normalized = gray_array.astype(float) / 255.0
            energy = np.sum(normalized**2)
            return float(energy / gray_array.size)
        except Exception as e:
            return 0.0
    
    def get_preset_params(self, preset_name):
        """Get parameters for a preset pattern"""
        return self.presets.get(preset_name, {})
    
    def generate_random_params(self):
        """Generate random parameters for pattern generation"""
        try:
            params = {
                'grid_size': random.choice([3, 5, 7, 9, 11]),
                'dot_density': random.uniform(0.4, 0.9),
                'symmetry_type': random.choice(['Bilateral', 'Rotational', 'Radial']),
                'complexity': random.choice(['Simple', 'Medium', 'Complex']),
                'curve_style': random.choice(['Smooth', 'Angular', 'Mixed', 'Flowing']),
                'theme': random.choice(['Geometric', 'Floral', 'Traditional', 'Modern']),
                'line_thickness': random.randint(2, 6),
                'smoothness': random.uniform(0.5, 2.0),
                'decorations': random.choice([True, False])
            }
            
            return params
            
        except Exception as e:
            logger.error(f"Error generating random parameters: {str(e)}")
            return {}
    
    def generate_surprise_params(self):
        """Generate surprising/unique parameters"""
        try:
            params = {
                'grid_size': random.choice([9, 11, 13, 15]),
                'dot_density': random.uniform(0.6, 1.0),
                'symmetry_type': 'Mixed',
                'complexity': 'Very Complex',
                'curve_style': 'Mixed',
                'theme': random.choice(['Modern', 'Abstract']),
                'line_thickness': random.randint(1, 8),
                'smoothness': random.uniform(0.8, 1.5),
                'decorations': True,
                'artistic_effects': True
            }
            
            return params
            
        except Exception as e:
            logger.error(f"Error generating surprise parameters: {str(e)}")
            return self.generate_random_params()
    
    def generate_sample(self, params):
        """Generate a small sample pattern for preview"""
        try:
            # Reduce size for sample
            sample_params = params.copy()
            sample_params['grid_size'] = min(5, params.get('grid_size', 5))
            
            result = self.generate_pattern(sample_params)
            if result and result['image']:
                # Resize for thumbnail
                sample_image = result['image'].resize((150, 150), Image.Resampling.LANCZOS)
                return sample_image
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating sample: {str(e)}")
            return None
    
    def export_to_svg(self, pattern_data):
        """Export pattern to SVG format"""
        try:
            # This is a simplified SVG export
            # In a full implementation, you would track drawing commands
            # and convert them to SVG paths
            
            svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="600" height="600" xmlns="http://www.w3.org/2000/svg">
    <rect width="600" height="600" fill="black"/>
    <!-- Pattern elements would be added here -->
    <text x="300" y="300" fill="white" text-anchor="middle">SVG Export</text>
</svg>'''
            
            return svg_content
            
        except Exception as e:
            logger.error(f"Error exporting to SVG: {str(e)}")
            return "<svg></svg>"
    
    def create_color_variation(self, base_pattern):
        """Create a color variation of existing pattern"""
        try:
            if not base_pattern or 'image' not in base_pattern:
                return None
            
            image = base_pattern['image'].copy()
            
            # Apply color transformation
            # Convert to HSV and modify hue
            hsv_image = image.convert('HSV')
            hsv_array = np.array(hsv_image)
            
            # Shift hue
            hue_shift = random.randint(30, 330)
            hsv_array[:, :, 0] = (hsv_array[:, :, 0] + hue_shift) % 360
            
            # Convert back to RGB
            new_image = Image.fromarray(hsv_array, 'HSV').convert('RGB')
            
            # Update pattern data
            new_pattern = base_pattern.copy()
            new_pattern['image'] = new_image
            
            return new_pattern
            
        except Exception as e:
            logger.error(f"Error creating color variation: {str(e)}")
            return base_pattern
    
    def create_size_variation(self, base_pattern):
        """Create a size variation of existing pattern"""
        try:
            if not base_pattern or 'image' not in base_pattern:
                return None
            
            # Scale the image
            original_image = base_pattern['image']
            scale_factor = random.uniform(0.7, 1.5)
            
            new_size = (
                int(original_image.width * scale_factor),
                int(original_image.height * scale_factor)
            )
            
            new_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Center on canvas of original size
            canvas = Image.new('RGB', original_image.size, 'black')
            paste_x = (canvas.width - new_image.width) // 2
            paste_y = (canvas.height - new_image.height) // 2
            canvas.paste(new_image, (paste_x, paste_y))
            
            # Update pattern data
            new_pattern = base_pattern.copy()
            new_pattern['image'] = canvas
            
            return new_pattern
            
        except Exception as e:
            logger.error(f"Error creating size variation: {str(e)}")
            return base_pattern
    
    def create_style_variation(self, base_pattern):
        """Create a style variation of existing pattern"""
        try:
            if not base_pattern or 'image' not in base_pattern:
                return None
            
            image = base_pattern['image'].copy()
            
            # Apply different artistic filters
            style_effects = [
                lambda img: img.filter(ImageFilter.BLUR),
                lambda img: img.filter(ImageFilter.EMBOSS),
                lambda img: img.filter(ImageFilter.EDGE_ENHANCE),
                lambda img: img.filter(ImageFilter.SMOOTH)
            ]
            
            effect = random.choice(style_effects)
            new_image = effect(image)
            
            # Update pattern data
            new_pattern = base_pattern.copy()
            new_pattern['image'] = new_image
            
            return new_pattern
            
        except Exception as e:
            logger.error(f"Error creating style variation: {str(e)}")
            return base_pattern
