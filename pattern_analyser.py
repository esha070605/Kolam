import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import math
import logging
from scipy import ndimage
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """
    OpenCV-based pattern analysis for Kolam patterns.
    Provides dot detection, symmetry analysis, and geometric feature extraction.
    """
    
    def __init__(self):
        self.debug = False
        logger.info("Pattern analyzer initialized")
    
    def analyze_pattern(self, image):
        """
        Comprehensive analysis of Kolam pattern
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array.copy()
            
            # Initialize results dictionary
            results = {
                'dot_count': 0,
                'contour_count': 0,
                'symmetry_score': 0.0,
                'complexity': 0.0,
                'pattern_type': 'Unknown',
                'grid_size': 'N/A',
                'features': {},
                'annotated_image': None
            }
            
            # Perform individual analyses
            dot_analysis = self.detect_dots(gray)
            contour_analysis = self.analyze_contours(gray)
            symmetry_analysis = self.analyze_symmetry(gray)
            geometric_features = self.extract_geometric_features(gray)
            
            # Combine results
            results.update(dot_analysis)
            results.update(contour_analysis)
            results.update(symmetry_analysis)
            results['features'] = geometric_features
            
            # Determine pattern type
            results['pattern_type'] = self.classify_pattern_type(results)
            
            # Create annotated image
            results['annotated_image'] = self.create_annotated_image(img_array, results)
            
            logger.info(f"Pattern analysis completed: {results['pattern_type']}")
            return results
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return {
                'error': str(e),
                'dot_count': 0,
                'contour_count': 0,
                'symmetry_score': 0.0,
                'complexity': 0.0,
                'pattern_type': 'Error',
                'grid_size': 'N/A',
                'features': {},
                'annotated_image': None
            }
    
    def detect_dots(self, gray_image):
        """
        Detect dots in the Kolam pattern using multiple methods
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary with dot detection results
        """
        try:
            results = {
                'dot_count': 0,
                'dots_detected': [],
                'grid_size': 'N/A',
                'dot_spacing': 0.0
            }
            
            # Method 1: HoughCircles for circular dots
            circles = cv2.HoughCircles(
                gray_image,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=3,
                maxRadius=20
            )
            
            dots_hough = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                dots_hough = [(x, y, r) for x, y, r in circles]
                results['dot_count'] += len(dots_hough)
            
            # Method 2: Blob detection for irregular dots
            # Create SimpleBlobDetector
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 10
            params.maxArea = 200
            params.filterByCircularity = True
            params.minCircularity = 0.1
            params.filterByConvexity = True
            params.minConvexity = 0.5
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray_image)
            
            dots_blob = [(int(kp.pt[0]), int(kp.pt[1]), int(kp.size/2)) for kp in keypoints]
            
            # Combine and deduplicate dots
            all_dots = dots_hough + dots_blob
            if all_dots:
                # Remove duplicates using clustering
                if len(all_dots) > 1:
                    points = np.array([(x, y) for x, y, r in all_dots])
                    clustering = DBSCAN(eps=15, min_samples=1).fit(points)
                    
                    unique_dots = []
                    for cluster_id in set(clustering.labels_):
                        cluster_points = points[clustering.labels_ == cluster_id]
                        center = np.mean(cluster_points, axis=0)
                        unique_dots.append((int(center[0]), int(center[1]), 5))
                    
                    results['dots_detected'] = unique_dots
                    results['dot_count'] = len(unique_dots)
                else:
                    results['dots_detected'] = all_dots
                    results['dot_count'] = len(all_dots)
                
                # Estimate grid properties
                if len(results['dots_detected']) >= 4:
                    grid_info = self.estimate_grid_properties(results['dots_detected'])
                    results.update(grid_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in dot detection: {str(e)}")
            return {
                'dot_count': 0,
                'dots_detected': [],
                'grid_size': 'N/A',
                'dot_spacing': 0.0
            }
    
    def estimate_grid_properties(self, dots):
        """
        Estimate grid size and spacing from detected dots
        
        Args:
            dots: List of detected dots [(x, y, r), ...]
            
        Returns:
            Dictionary with grid properties
        """
        try:
            if len(dots) < 4:
                return {'grid_size': 'N/A', 'dot_spacing': 0.0}
            
            # Extract coordinates
            coords = np.array([(x, y) for x, y, r in dots])
            
            # Find approximate grid dimensions
            x_coords = sorted(set(coords[:, 0]))
            y_coords = sorted(set(coords[:, 1]))
            
            # Estimate spacing
            if len(x_coords) > 1:
                x_spacings = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
                avg_x_spacing = np.mean(x_spacings)
            else:
                avg_x_spacing = 0
            
            if len(y_coords) > 1:
                y_spacings = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
                avg_y_spacing = np.mean(y_spacings)
            else:
                avg_y_spacing = 0
            
            avg_spacing = (avg_x_spacing + avg_y_spacing) / 2
            
            # Estimate grid size
            if avg_spacing > 0:
                grid_width = int(round((max(x_coords) - min(x_coords)) / avg_spacing)) + 1
                grid_height = int(round((max(y_coords) - min(y_coords)) / avg_spacing)) + 1
                grid_size = f"{grid_width}×{grid_height}"
            else:
                grid_size = "Unknown"
            
            return {
                'grid_size': grid_size,
                'dot_spacing': float(avg_spacing)
            }
            
        except Exception as e:
            logger.error(f"Error estimating grid properties: {str(e)}")
            return {'grid_size': 'N/A', 'dot_spacing': 0.0}
    
    def analyze_contours(self, gray_image):
        """
        Analyze contours and shapes in the pattern
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary with contour analysis results
        """
        try:
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            min_area = 100
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            results = {
                'contour_count': len(filtered_contours),
                'total_contours': len(contours),
                'shapes_detected': [],
                'total_area': 0.0,
                'perimeter_total': 0.0
            }
            
            for contour in filtered_contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                results['total_area'] += area
                results['perimeter_total'] += perimeter
                
                # Approximate contour to detect shapes
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Classify shape based on number of vertices
                vertices = len(approx)
                if vertices == 3:
                    shape = "Triangle"
                elif vertices == 4:
                    shape = "Rectangle/Square"
                elif vertices > 4 and vertices < 10:
                    shape = f"Polygon ({vertices} sides)"
                else:
                    shape = "Circle/Curve"
                
                results['shapes_detected'].append({
                    'shape': shape,
                    'area': float(area),
                    'perimeter': float(perimeter),
                    'vertices': vertices
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in contour analysis: {str(e)}")
            return {
                'contour_count': 0,
                'total_contours': 0,
                'shapes_detected': [],
                'total_area': 0.0,
                'perimeter_total': 0.0
            }
    
    def analyze_symmetry(self, gray_image):
        """
        Analyze symmetry properties of the pattern
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary with symmetry analysis results
        """
        try:
            height, width = gray_image.shape
            
            # Bilateral symmetry (vertical axis)
            left_half = gray_image[:, :width//2]
            right_half = gray_image[:, width//2:]
            right_half_flipped = np.fliplr(right_half)
            
            # Resize to match if different sizes
            if left_half.shape[1] != right_half_flipped.shape[1]:
                min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_half_flipped = right_half_flipped[:, :min_width]
            
            bilateral_symmetry = self.calculate_image_similarity(left_half, right_half_flipped)
            
            # Bilateral symmetry (horizontal axis)
            top_half = gray_image[:height//2, :]
            bottom_half = gray_image[height//2:, :]
            bottom_half_flipped = np.flipud(bottom_half)
            
            if top_half.shape[0] != bottom_half_flipped.shape[0]:
                min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
                top_half = top_half[:min_height, :]
                bottom_half_flipped = bottom_half_flipped[:min_height, :]
            
            horizontal_bilateral = self.calculate_image_similarity(top_half, bottom_half_flipped)
            
            # Rotational symmetry (90, 180, 270 degrees)
            rotational_90 = self.calculate_rotational_symmetry(gray_image, 90)
            rotational_180 = self.calculate_rotational_symmetry(gray_image, 180)
            rotational_270 = self.calculate_rotational_symmetry(gray_image, 270)
            
            # Overall symmetry score
            symmetry_score = np.mean([
                bilateral_symmetry,
                horizontal_bilateral,
                rotational_180
            ])
            
            results = {
                'bilateral_symmetry': float(bilateral_symmetry),
                'horizontal_bilateral_symmetry': float(horizontal_bilateral),
                'rotational_symmetry': float(np.mean([rotational_90, rotational_180, rotational_270])),
                'rotational_90': float(rotational_90),
                'rotational_180': float(rotational_180),
                'rotational_270': float(rotational_270),
                'symmetry_score': float(symmetry_score),
                'translational_symmetry': 0.0  # Placeholder for future implementation
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in symmetry analysis: {str(e)}")
            return {
                'bilateral_symmetry': 0.0,
                'horizontal_bilateral_symmetry': 0.0,
                'rotational_symmetry': 0.0,
                'rotational_90': 0.0,
                'rotational_180': 0.0,
                'rotational_270': 0.0,
                'symmetry_score': 0.0,
                'translational_symmetry': 0.0
            }
    
    def calculate_image_similarity(self, img1, img2):
        """
        Calculate similarity between two images using normalized cross-correlation
        
        Args:
            img1, img2: Images to compare
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Ensure same size
            if img1.shape != img2.shape:
                min_h = min(img1.shape[0], img2.shape[0])
                min_w = min(img1.shape[1], img2.shape[1])
                img1 = img1[:min_h, :min_w]
                img2 = img2[:min_h, :min_w]
            
            # Normalize images
            img1_norm = img1.astype(np.float32) / 255.0
            img2_norm = img2.astype(np.float32) / 255.0
            
            # Calculate normalized cross-correlation
            correlation = cv2.matchTemplate(img1_norm, img2_norm, cv2.TM_CCOEFF_NORMED)
            similarity = float(correlation[0, 0])
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return max(0, min(1, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating image similarity: {str(e)}")
            return 0.0
    
    def calculate_rotational_symmetry(self, image, angle):
        """
        Calculate rotational symmetry for given angle
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Symmetry score (0-1)
        """
        try:
            height, width = image.shape
            center = (width // 2, height // 2)
            
            # Rotate image
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            # Calculate similarity with original
            similarity = self.calculate_image_similarity(image, rotated)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating rotational symmetry: {str(e)}")
            return 0.0
    
    def extract_geometric_features(self, gray_image):
        """
        Extract various geometric features from the pattern
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary with geometric features
        """
        try:
            features = {}
            
            # Apply threshold
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate basic properties
            total_pixels = binary.size
            white_pixels = np.sum(binary == 255)
            black_pixels = np.sum(binary == 0)
            
            features['area_coverage'] = float(white_pixels / total_pixels)
            features['background_ratio'] = float(black_pixels / total_pixels)
            
            # Calculate fractal dimension using box-counting method
            features['fractal_dimension'] = self.calculate_fractal_dimension(binary)
            
            # Calculate complexity based on edge density
            edges = cv2.Canny(gray_image, 50, 150)
            edge_pixels = np.sum(edges > 0)
            features['edge_density'] = float(edge_pixels / total_pixels)
            
            # Calculate perimeter to area ratio
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                total_area = sum(cv2.contourArea(cnt) for cnt in contours)
                total_perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
                if total_area > 0:
                    features['perimeter_ratio'] = float(total_perimeter / total_area)
                else:
                    features['perimeter_ratio'] = 0.0
            else:
                features['perimeter_ratio'] = 0.0
            
            # Calculate moments and shape descriptors
            moments = cv2.moments(binary)
            if moments['m00'] != 0:
                features['centroid_x'] = float(moments['m10'] / moments['m00'])
                features['centroid_y'] = float(moments['m01'] / moments['m00'])
                features['hu_moments'] = [float(hu) for hu in cv2.HuMoments(moments).flatten()]
            else:
                features['centroid_x'] = 0.0
                features['centroid_y'] = 0.0
                features['hu_moments'] = [0.0] * 7
            
            # Calculate texture features using Local Binary Patterns
            features.update(self.calculate_texture_features(gray_image))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting geometric features: {str(e)}")
            return {}
    
    def calculate_fractal_dimension(self, binary_image):
        """
        Calculate fractal dimension using box-counting method
        
        Args:
            binary_image: Binary image
            
        Returns:
            Fractal dimension
        """
        try:
            # Convert to binary if needed
            if len(binary_image.shape) == 3:
                binary_image = cv2.cvtColor(binary_image, cv2.COLOR_RGB2GRAY)
            
            binary_image = (binary_image > 0).astype(np.uint8)
            
            # Box sizes
            sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25]
            counts = []
            
            for size in sizes:
                # Count boxes containing pattern
                count = 0
                for i in range(0, binary_image.shape[0], size):
                    for j in range(0, binary_image.shape[1], size):
                        box = binary_image[i:i+size, j:j+size]
                        if np.any(box):
                            count += 1
                counts.append(count)
            
            # Calculate fractal dimension
            if len(counts) > 1 and all(c > 0 for c in counts):
                coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                fractal_dim = -coeffs[0]
                return max(1.0, min(2.0, fractal_dim))
            else:
                return 1.5
                
        except Exception as e:
            logger.error(f"Error calculating fractal dimension: {str(e)}")
            return 1.5
    
    def calculate_texture_features(self, gray_image):
        """
        Calculate texture features
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary with texture features
        """
        try:
            features = {}
            
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features['gradient_mean'] = float(np.mean(gradient_magnitude))
            features['gradient_std'] = float(np.std(gradient_magnitude))
            
            # Calculate local standard deviation
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_image.astype(np.float32) - local_mean)**2, -1, kernel)
            local_std = np.sqrt(local_variance)
            
            features['local_std_mean'] = float(np.mean(local_std))
            features['local_std_max'] = float(np.max(local_std))
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating texture features: {str(e)}")
            return {}
    
    def calculate_mathematical_metrics(self, image):
        """
        Calculate comprehensive mathematical metrics
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with mathematical metrics
        """
        try:
            # Convert to grayscale if needed
            if isinstance(image, Image.Image):
                gray = np.array(image.convert('L'))
            elif len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Analyze the pattern
            analysis_results = self.analyze_pattern(image)
            
            # Extract metrics from analysis
            metrics = {
                'bilateral_symmetry': analysis_results.get('bilateral_symmetry', 0.0),
                'rotational_symmetry': analysis_results.get('rotational_symmetry', 0.0),
                'translational_symmetry': analysis_results.get('translational_symmetry', 0.0),
                'fractal_dimension': analysis_results.get('features', {}).get('fractal_dimension', 1.5),
                'perimeter_ratio': analysis_results.get('features', {}).get('perimeter_ratio', 0.0),
                'area_coverage': analysis_results.get('features', {}).get('area_coverage', 0.0),
                'edge_density': analysis_results.get('features', {}).get('edge_density', 0.0),
                'complexity_score': self.calculate_complexity_score(analysis_results),
                'geometric_regularity': self.calculate_geometric_regularity(gray),
                'pattern_coherence': self.calculate_pattern_coherence(gray)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating mathematical metrics: {str(e)}")
            return {}
    
    def calculate_complexity_score(self, analysis_results):
        """
        Calculate overall complexity score of the pattern
        
        Args:
            analysis_results: Results from pattern analysis
            
        Returns:
            Complexity score (0-1)
        """
        try:
            factors = []
            
            # Dot count contribution
            dot_count = analysis_results.get('dot_count', 0)
            if dot_count > 0:
                factors.append(min(1.0, dot_count / 50.0))
            
            # Contour count contribution
            contour_count = analysis_results.get('contour_count', 0)
            if contour_count > 0:
                factors.append(min(1.0, contour_count / 20.0))
            
            # Edge density contribution
            edge_density = analysis_results.get('features', {}).get('edge_density', 0.0)
            factors.append(edge_density)
            
            # Symmetry complexity (higher symmetry = lower complexity)
            symmetry_score = analysis_results.get('symmetry_score', 0.0)
            factors.append(1.0 - symmetry_score)
            
            if factors:
                complexity = np.mean(factors)
                return float(max(0.0, min(1.0, complexity)))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating complexity score: {str(e)}")
            return 0.0
    
    def calculate_geometric_regularity(self, gray_image):
        """
        Calculate how regular/consistent the geometric patterns are
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Regularity score (0-1)
        """
        try:
            # Find contours
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) < 2:
                return 0.0
            
            # Calculate area variations
            areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]
            
            if len(areas) < 2:
                return 0.0
            
            # Calculate coefficient of variation
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            
            if mean_area > 0:
                cv_area = std_area / mean_area
                regularity = max(0.0, 1.0 - cv_area)
                return float(regularity)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating geometric regularity: {str(e)}")
            return 0.0
    
    def calculate_pattern_coherence(self, gray_image):
        """
        Calculate how well the pattern elements work together
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Coherence score (0-1)
        """
        try:
            # Calculate global and local statistics
            global_mean = np.mean(gray_image)
            global_std = np.std(gray_image)
            
            # Calculate local coherence using sliding window
            kernel_size = 15
            local_means = []
            local_stds = []
            
            for i in range(0, gray_image.shape[0] - kernel_size, kernel_size//2):
                for j in range(0, gray_image.shape[1] - kernel_size, kernel_size//2):
                    patch = gray_image[i:i+kernel_size, j:j+kernel_size]
                    local_means.append(np.mean(patch))
                    local_stds.append(np.std(patch))
            
            if local_means and local_stds:
                # Calculate consistency of local statistics
                mean_consistency = 1.0 - (np.std(local_means) / (global_mean + 1e-6))
                std_consistency = 1.0 - (np.std(local_stds) / (global_std + 1e-6))
                
                coherence = (mean_consistency + std_consistency) / 2
                return float(max(0.0, min(1.0, coherence)))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating pattern coherence: {str(e)}")
            return 0.0
    
    def classify_pattern_type(self, analysis_results):
        """
        Classify the pattern type based on analysis results
        
        Args:
            analysis_results: Dictionary with analysis results
            
        Returns:
            Pattern type string
        """
        try:
            dot_count = analysis_results.get('dot_count', 0)
            contour_count = analysis_results.get('contour_count', 0)
            symmetry_score = analysis_results.get('symmetry_score', 0.0)
            
            # Classification logic
            if dot_count > 10 and symmetry_score > 0.7:
                return "Traditional Pulli"
            elif contour_count > 15 and symmetry_score > 0.5:
                return "Complex Geometric"
            elif symmetry_score > 0.8:
                return "Symmetric Pattern"
            elif contour_count > 5:
                return "Geometric Pattern"
            elif dot_count > 0:
                return "Dot-based Pattern"
            else:
                return "Simple Pattern"
                
        except Exception as e:
            logger.error(f"Error classifying pattern type: {str(e)}")
            return "Unknown"
    
    def create_annotated_image(self, original_image, analysis_results):
        """
        Create annotated image showing analysis results
        
        Args:
            original_image: Original image
            analysis_results: Analysis results dictionary
            
        Returns:
            Annotated PIL Image
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(original_image, np.ndarray):
                if len(original_image.shape) == 2:
                    img = Image.fromarray(original_image, mode='L').convert('RGB')
                else:
                    img = Image.fromarray(original_image)
            else:
                img = original_image.copy()
            
            # Create drawing object
            draw = ImageDraw.Draw(img)
            
            # Draw detected dots
            dots = analysis_results.get('dots_detected', [])
            for x, y, r in dots:
                draw.ellipse([x-r, y-r, x+r, y+r], outline='red', width=2)
                draw.point((x, y), fill='red')
            
            # Add text annotations
            try:
                # Use default font
                font = None
                
                # Add analysis information
                info_text = [
                    f"Dots: {analysis_results.get('dot_count', 0)}",
                    f"Contours: {analysis_results.get('contour_count', 0)}",
                    f"Symmetry: {analysis_results.get('symmetry_score', 0.0):.2f}",
                    f"Type: {analysis_results.get('pattern_type', 'Unknown')}"
                ]
                
                y_offset = 10
                for text in info_text:
                    draw.text((10, y_offset), text, fill='blue', font=font)
                    y_offset += 20
                    
            except Exception as e:
                logger.warning(f"Could not add text annotations: {str(e)}")
            
            return img
            
        except Exception as e:
            logger.error(f"Error creating annotated image: {str(e)}")
            return original_image
