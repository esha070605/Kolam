import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KolamCNN:
    """
    Convolutional Neural Network for Kolam pattern classification.
    Classifies patterns into: Geometric, Floral, Animal, Traditional, Modern
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.num_classes = 5
        self.class_names = ['Geometric', 'Floral', 'Animal', 'Traditional', 'Modern']
        self.model_path = model_path or 'models/kolam_cnn_model.h5'
        
        # Try to load existing model, otherwise create new one
        try:
            if os.path.exists(self.model_path):
                self.load_model()
                logger.info("Loaded existing CNN model")
            else:
                self.build_model()
                logger.info("Built new CNN model")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self.build_model()
    
    def build_model(self):
        """
        Build CNN architecture with attention mechanisms for Kolam classification
        """
        try:
            # Input layer
            inputs = keras.Input(shape=self.input_shape)
            
            # Data augmentation layer
            data_augmentation = keras.Sequential([
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomFlip("horizontal"),
                layers.RandomContrast(0.1),
            ])
            
            x = data_augmentation(inputs)
            
            # Preprocessing
            x = layers.Rescaling(1./255)(x)
            
            # First convolutional block
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.25)(x)
            
            # Second convolutional block
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.25)(x)
            
            # Third convolutional block
            x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.25)(x)
            
            # Fourth convolutional block with attention
            x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            
            # Attention mechanism
            attention = layers.GlobalAveragePooling2D()(x)
            attention = layers.Dense(256, activation='relu')(attention)
            attention = layers.Dense(256, activation='sigmoid')(attention)
            attention = layers.Reshape((1, 1, 256))(attention)
            x = layers.multiply([x, attention])
            
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.25)(x)
            
            # Global average pooling and dense layers
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            
            # Output layer
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            # Create model
            self.model = keras.Model(inputs, outputs)
            
            # Compile model
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("CNN model built successfully")
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess image for CNN prediction
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image ready for prediction
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize to input shape
            image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
            
            # Ensure correct data type
            image = image.astype(np.float32)
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image):
        """
        Predict Kolam pattern category
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return {
                    'predicted_class': 'Unknown',
                    'confidence': 0.0,
                    'probabilities': [0.2] * self.num_classes,
                    'error': 'Model not loaded'
                }
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            probabilities = predictions[0]
            
            # Get predicted class
            predicted_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Convert probabilities to list for JSON serialization
            probabilities_list = [float(p) for p in probabilities]
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities_list,
                'class_names': self.class_names
            }
            
            logger.info(f"Prediction: {predicted_class} ({confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'predicted_class': 'Error',
                'confidence': 0.0,
                'probabilities': [0.2] * self.num_classes,
                'class_names': self.class_names,
                'error': str(e)
            }
    
    def load_model(self):
        """Load saved model from file"""
        try:
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def save_model(self):
        """Save model to file"""
        try:
            if self.model is not None:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save(self.model_path)
                logger.info(f"Model saved to {self.model_path}")
            else:
                logger.error("No model to save")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def train(self, train_data, validation_data, epochs=50):
        """
        Train the CNN model
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of training epochs
        """
        try:
            if self.model is None:
                self.build_model()
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True,
                    monitor='val_accuracy'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.2,
                    patience=5,
                    monitor='val_accuracy'
                ),
                keras.callbacks.ModelCheckpoint(
                    self.model_path,
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
            
            # Train model
            history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Training completed")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def evaluate_model(self, test_data):
        """
        Evaluate model performance
        
        Args:
            test_data: Test dataset
            
        Returns:
            Evaluation metrics
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            results = self.model.evaluate(test_data, verbose=0)
            
            metrics = {
                'loss': float(results[0]),
                'accuracy': float(results[1])
            }
            
            logger.info(f"Evaluation results: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return None
    
    def get_model_summary(self):
        """Get model architecture summary"""
        try:
            if self.model is not None:
                return self.model.summary()
            else:
                return "Model not loaded"
        except Exception as e:
            logger.error(f"Error getting model summary: {str(e)}")
            return f"Error: {str(e)}"
    
    def predict_batch(self, images):
        """
        Predict multiple images at once
        
        Args:
            images: List of images
            
        Returns:
            List of prediction results
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return []
            
            results = []
            for image in images:
                result = self.predict(image)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            return []
    
    def get_feature_maps(self, image, layer_name=None):
        """
        Extract feature maps from intermediate layers
        
        Args:
            image: Input image
            layer_name: Name of layer to extract features from
            
        Returns:
            Feature maps
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Create feature extraction model
            if layer_name:
                layer = self.model.get_layer(layer_name)
                feature_model = keras.Model(inputs=self.model.input, outputs=layer.output)
            else:
                # Get last convolutional layer
                conv_layers = [layer for layer in self.model.layers if 'conv' in layer.name.lower()]
                if conv_layers:
                    feature_model = keras.Model(inputs=self.model.input, outputs=conv_layers[-1].output)
                else:
                    logger.error("No convolutional layers found")
                    return None
            
            # Extract features
            features = feature_model.predict(processed_image, verbose=0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting feature maps: {str(e)}")
            return None
