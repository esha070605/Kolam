import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image, resize=True, enhance_contrast=True, noise_reduction=False, target_size=(512, 512)):
    """
    Preprocess image for analysis
    
    Args:
        image: PIL Image or numpy array
        resize: Whether to resize image
        enhance_contrast: Whether to enhance contrast
        noise_reduction: Whether to apply noise reduction
        target_size: Target size for resize
        
    Returns:
        Preprocessed PIL Image
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                img = Image.fromarray(image, mode='L').convert('RGB')
            else:
                img = Image.fromarray(image)
        else:
            img = image.copy()
        
        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if requested
        if resize:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            logger.info(f"Image resized to {target_size}")
        
        # Enhance contrast if requested
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            logger.info("Contrast enhanced")
        
        # Apply noise reduction if requested
        if noise_reduction:
            # Convert to numpy for OpenCV operations
            img_array = np.array(img)
            
            # Apply bilateral filter for noise reduction
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
            
            # Convert back to PIL
            img = Image.fromarray(img_array)
            logger.info("Noise reduction applied")
        
        return img
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return image

def apply_color_scheme(image, color_scheme, custom_colors=None):
    """
    Apply color scheme to pattern image
    
    Args:
        image: PIL Image
        color_scheme: Color scheme name
        custom_colors: List of custom colors for 'Custom' scheme
        
    Returns:
        PIL Image with applied color scheme
    """
    try:
        img = image.copy()
        
        if color_scheme == "Traditional White":
            # Keep original (white on black)
            return img
        
        elif color_scheme == "Vibrant Colors":
            # Apply vibrant color transformation
            img_array = np.array(img)
            
            # Create a colorful version
            # Replace white pixels with vibrant colors
            mask = np.all(img_array > [200, 200, 200], axis=2)
            
            # Apply gradient colors
            height, width = mask.shape
            for i in range(height):
                color_ratio = i / height
                color = [
                    int(255 * (1 - color_ratio) + 255 * color_ratio * 0.8),  # Red
                    int(100 * color_ratio + 150),  # Green
                    int(50 + 205 * color_ratio)    # Blue
                ]
                img_array[mask & (np.arange(height)[:, None] == i)] = color
            
            return Image.fromarray(img_array)
        
        elif color_scheme == "Pastel":
            # Apply pastel colors
            img_array = np.array(img)
            mask = np.all(img_array > [200, 200, 200], axis=2)
            
            # Soft pastel colors
            pastel_colors = [
                [255, 182, 193],  # Light pink
                [173, 216, 230],  # Light blue
                [144, 238, 144],  # Light green
                [255, 218, 185],  # Peach
                [221, 160, 221]   # Plum
            ]
            
            # Apply random pastel color to different regions
            for i, color in enumerate(pastel_colors):
                region_mask = mask & ((np.arange(img_array.shape[0])[:, None] % len(pastel_colors)) == i)
                img_array[region_mask] = color
            
            return Image.fromarray(img_array)
        
        elif color_scheme == "Monochrome":
            # Convert to grayscale with blue tint
            gray = img.convert('L')
            img_array = np.array(gray)
            
            # Create blue monochrome
            blue_mono = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
            blue_mono[:, :, 2] = img_array  # Blue channel
            blue_mono[:, :, 0] = img_array // 4  # Slight red
            blue_mono[:, :, 1] = img_array // 2  # Some green
            
            return Image.fromarray(blue_mono)
        
        elif color_scheme == "Custom" and custom_colors:
            # Apply custom colors
            img_array = np.array(img)
            mask = np.all(img_array > [200, 200, 200], axis=2)
            
            # Convert hex colors to RGB if needed
            rgb_colors = []
            for color in custom_colors:
                if isinstance(color, str) and color.startswith('#'):
                    # Convert hex to RGB
                    color = color.lstrip('#')
                    rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                    rgb_colors.append(list(rgb))
                else:
                    rgb_colors.append(color)
            
            if rgb_colors:
                # Apply primary custom color
                img_array[mask] = rgb_colors[0]
            
            return Image.fromarray(img_array)
        
        return img
        
    except Exception as e:
        logger.error(f"Error applying color scheme: {str(e)}")
        return image

def add_artistic_effects(image):
    """
    Add artistic effects to the image
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image with artistic effects
    """
    try:
        img = image.copy()
        
        # Apply a combination of artistic filters
        
        # 1. Slight blur for softer look
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # 2. Enhance sharpness slightly
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
        
        # 3. Adjust brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        
        # 4. Add subtle emboss effect
        emboss_filter = ImageFilter.EMBOSS
        embossed = img.filter(emboss_filter)
        
        # Blend original with embossed
        img = Image.blend(img, embossed, 0.1)
        
        return img
        
    except Exception as e:
        logger.error(f"Error adding artistic effects: {str(e)}")
        return image

def canvas_to_kolam(canvas_data):
    """
    Convert drawing canvas data to Kolam image
    
    Args:
        canvas_data: Canvas image data (numpy array)
        
    Returns:
        PIL Image
    """
    try:
        # Convert canvas data to PIL Image
        if isinstance(canvas_data, np.ndarray):
            # Handle different array shapes
            if len(canvas_data.shape) == 4:  # RGBA with batch dimension
                canvas_data = canvas_data[0]  # Remove batch dimension
            
            if canvas_data.shape[2] == 4:  # RGBA
                img = Image.fromarray(canvas_data.astype(np.uint8), 'RGBA')
                # Convert to RGB with black background
                background = Image.new('RGB', img.size, (0, 0, 0))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
            else:  # RGB
                img = Image.fromarray(canvas_data.astype(np.uint8), 'RGB')
        else:
            img = canvas_data
        
        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        logger.error(f"Error converting canvas to Kolam: {str(e)}")
        # Return a default black image
        return Image.new('RGB', (600, 600), (0, 0, 0))

def add_grid_overlay(canvas_size, grid_size, opacity=0.5):
    """
    Create grid overlay for drawing assistance
    
    Args:
        canvas_size: Size of the canvas
        grid_size: Grid dimensions (NxN)
        opacity: Grid opacity (0-1)
        
    Returns:
        PIL Image with grid overlay
    """
    try:
        # Create transparent image for grid
        grid_img = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(grid_img)
        
        # Calculate grid spacing
        spacing = canvas_size / grid_size
        
        # Draw grid dots
        dot_radius = 3
        dot_color = (255, 255, 255, int(255 * opacity))
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = int(i * spacing + spacing / 2)
                y = int(j * spacing + spacing / 2)
                
                # Draw dot
                draw.ellipse([
                    x - dot_radius, y - dot_radius,
                    x + dot_radius, y + dot_radius
                ], fill=dot_color)
        
        # Draw grid lines (optional, lighter)
        line_color = (255, 255, 255, int(255 * opacity * 0.3))
        
        for i in range(grid_size + 1):
            x = int(i * spacing)
            y = int(i * spacing)
            
            # Vertical line
            draw.line([(x, 0), (x, canvas_size)], fill=line_color, width=1)
            
            # Horizontal line  
            draw.line([(0, y), (canvas_size, y)], fill=line_color, width=1)
        
        return grid_img
        
    except Exception as e:
        logger.error(f"Error creating grid overlay: {str(e)}")
        return Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))

def validate_kolam_rules(image, continuous_line=True):
    """
    Validate if the pattern follows traditional Kolam rules
    
    Args:
        image: PIL Image or numpy array
        continuous_line: Whether to check for continuous line rule
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Convert to grayscale for analysis
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        elif len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        violations = []
        score = 1.0
        
        # Rule 1: Pattern should have dots or connection points
        # Detect circular features (dots)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=2,
            maxRadius=15
        )
        
        has_dots = circles is not None and len(circles[0]) > 0
        if not has_dots:
            violations.append("No clear dot structure detected")
            score -= 0.2
        
        # Rule 2: Check for symmetry (traditional Kolams are often symmetric)
        symmetry_score = calculate_symmetry_score(gray)
        if symmetry_score < 0.3:
            violations.append("Low symmetry detected")
            score -= 0.3
        
        # Rule 3: Check for continuous lines (if requested)
        continuity_score = 1.0
        if continuous_line:
            continuity_score = check_line_continuity(gray)
            if continuity_score < 0.5:
                violations.append("Pattern may not follow continuous line rule")
                score -= 0.2
        
        # Rule 4: Check for appropriate complexity
        edge_density = calculate_edge_density(gray)
        if edge_density > 0.8:
            violations.append("Pattern may be overly complex")
            score -= 0.1
        elif edge_density < 0.1:
            violations.append("Pattern may be too simple")
            score -= 0.1
        
        # Rule 5: Check for closed loops
        has_loops = check_for_loops(gray)
        if not has_loops:
            violations.append("No closed loops detected")
            score -= 0.2
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return {
            'valid': len(violations) == 0,
            'score': score,
            'violations': violations,
            'symmetry_score': symmetry_score,
            'continuity_score': continuity_score if continuous_line else 1.0,
            'edge_density': edge_density,
            'has_dots': has_dots,
            'has_loops': has_loops
        }
        
    except Exception as e:
        logger.error(f"Error validating Kolam rules: {str(e)}")
        return {
            'valid': False,
            'score': 0.0,
            'violations': [f"Validation error: {str(e)}"],
            'symmetry_score': 0.0,
            'continuity_score': 0.0,
            'edge_density': 0.0,
            'has_dots': False,
            'has_loops': False
        }

def calculate_symmetry_score(gray_image):
    """Calculate symmetry score for the image"""
    try:
        height, width = gray_image.shape
        
        # Bilateral symmetry (vertical axis)
        left_half = gray_image[:, :width//2]
        right_half = gray_image[:, width//2:]
        right_flipped = np.fliplr(right_half)
        
        # Ensure same size
        min_width = min(left_half.shape[1], right_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_flipped = right_flipped[:, :min_width]
        
        # Calculate similarity
        diff = np.abs(left_half.astype(float) - right_flipped.astype(float))
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return max(0.0, similarity)
        
    except Exception as e:
        logger.error(f"Error calculating symmetry score: {str(e)}")
        return 0.0

def check_line_continuity(gray_image):
    """Check if lines in the pattern are continuous"""
    try:
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        
        # Skeletonize to get line structure
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary > 0)
        
        # Count junction points (where lines meet)
        # In a continuous line pattern, there should be minimal junctions
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        junctions = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        junction_count = np.sum(junctions > 2)  # Points with more than 2 neighbors
        
        total_line_pixels = np.sum(skeleton)
        
        if total_line_pixels > 0:
            continuity_score = 1.0 - (junction_count / total_line_pixels)
            return max(0.0, continuity_score)
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Error checking line continuity: {str(e)}")
        return 0.0

def calculate_edge_density(gray_image):
    """Calculate edge density in the image"""
    try:
        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Calculate density
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        
        density = edge_pixels / total_pixels
        return density
        
    except Exception as e:
        logger.error(f"Error calculating edge density: {str(e)}")
        return 0.0

def check_for_loops(gray_image):
    """Check if the pattern contains closed loops"""
    try:
        # Apply threshold
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for closed contours with reasonable area
        min_area = 100
        closed_loops = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return len(closed_loops) > 0
        
    except Exception as e:
        logger.error(f"Error checking for loops: {str(e)}")
        return False

def create_analysis_visualization(image, analysis_results, classification_results):
    """
    Create comprehensive visualization of analysis results
    
    Args:
        image: Original image
        analysis_results: Pattern analysis results
        classification_results: CNN classification results
        
    Returns:
        Dictionary with different visualizations
    """
    try:
        # Convert to grayscale for processing
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        elif len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        visualizations = {}
        
        # Edge detection visualization
        edges = cv2.Canny(gray, 50, 150)
        edge_colored = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        visualizations['Edge Detection'] = Image.fromarray(cv2.cvtColor(edge_colored, cv2.COLOR_BGR2RGB))
        
        # Contour analysis visualization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_img = np.zeros_like(gray)
        cv2.drawContours(contour_img, contours, -1, 255, 2)
        contour_colored = cv2.applyColorMap(contour_img, cv2.COLORMAP_RAINBOW)
        visualizations['Contour Analysis'] = Image.fromarray(cv2.cvtColor(contour_colored, cv2.COLOR_BGR2RGB))
        
        # Symmetry map
        symmetry_map = create_symmetry_map(gray)
        visualizations['Symmetry Map'] = symmetry_map
        
        # Feature heatmap
        feature_heatmap = create_feature_heatmap(gray)
        visualizations['Feature Heatmap'] = feature_heatmap
        
        # Add plot data for interactive visualization
        if analysis_results:
            visualizations['plot_data'] = {
                'x': list(range(len(analysis_results.get('features', {})))),
                'y': list(analysis_results.get('features', {}).values())
            }
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Error creating analysis visualization: {str(e)}")
        return {}

def create_symmetry_map(gray_image):
    """Create a visualization showing symmetry analysis"""
    try:
        height, width = gray_image.shape
        
        # Create symmetry map
        symmetry_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Bilateral symmetry visualization
        for y in range(height):
            for x in range(width):
                if x < width // 2:
                    # Left side - compare with right side
                    mirror_x = width - 1 - x
                    if mirror_x < width:
                        diff = abs(int(gray_image[y, x]) - int(gray_image[y, mirror_x]))
                        similarity = 255 - diff
                        symmetry_map[y, x] = [similarity, 0, 0]  # Red channel for bilateral
                else:
                    # Right side - copy from left analysis
                    mirror_x = width - 1 - x
                    if mirror_x >= 0:
                        symmetry_map[y, x] = symmetry_map[y, mirror_x]
        
        return Image.fromarray(symmetry_map)
        
    except Exception as e:
        logger.error(f"Error creating symmetry map: {str(e)}")
        return Image.new('RGB', (300, 300), (0, 0, 0))

def create_feature_heatmap(gray_image):
    """Create a heatmap showing feature density"""
    try:
        # Calculate local feature density using gradient magnitude
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_HOT)
        
        return Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        
    except Exception as e:
        logger.error(f"Error creating feature heatmap: {str(e)}")
        return Image.new('RGB', (300, 300), (0, 0, 0))

def enhance_image_quality(image):
    """
    Enhance image quality for better analysis
    
    Args:
        image: PIL Image
        
    Returns:
        Enhanced PIL Image
    """
    try:
        img = image.copy()
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        # Reduce noise while preserving edges
        img_array = np.array(img)
        
        # Apply bilateral filter if image is colored
        if len(img_array.shape) == 3:
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        else:
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        return Image.fromarray(img_array)
        
    except Exception as e:
        logger.error(f"Error enhancing image quality: {str(e)}")
        return image

def convert_to_analysis_format(image):
    """
    Convert image to optimal format for analysis
    
    Args:
        image: Input image (various formats)
        
    Returns:
        Processed numpy array suitable for analysis
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                img = Image.fromarray(image, mode='L')
            else:
                img = Image.fromarray(image)
        else:
            img = image.copy()
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to standard size for consistent analysis
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error converting to analysis format: {str(e)}")
        # Return default array
        return np.zeros((512, 512, 3), dtype=np.uint8)

def save_analysis_results(image, results, filename):
    """
    Save analysis results with visualization
    
    Args:
        image: Original image
        results: Analysis results dictionary
        filename: Output filename
    """
    try:
        # Create a composite image with original and analysis
        original = image.copy()
        
        # Create text overlay with results
        draw = ImageDraw.Draw(original)
        
        # Add key metrics as text
        y_offset = 10
        text_items = [
            f"Pattern Type: {results.get('pattern_type', 'Unknown')}",
            f"Dots: {results.get('dot_count', 0)}",
            f"Symmetry: {results.get('symmetry_score', 0.0):.2f}",
            f"Complexity: {results.get('complexity', 0.0):.2f}"
        ]
        
        for text in text_items:
            draw.text((10, y_offset), text, fill='white')
            y_offset += 25
        
        # Save the annotated image
        original.save(filename)
        logger.info(f"Analysis results saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
