"""
Data preprocessing utilities for Vision AI project.
Handles image loading, normalization, and augmentation.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import os


class DataPreprocessor:
    """Class for handling all data preprocessing operations."""
    
    def _init_(self, dataset_name: str = "cifar10"):
        """
        Initialize the data preprocessor.
        
        Args:
            dataset_name (str): Name of the dataset to load
        """
        self.dataset_name = dataset_name
        self.class_names = []
        self.input_shape = None
        
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and return the specified dataset.
        
        Returns:
            Tuple of (x_train, y_train, x_test, y_test)
        """
        if self.dataset_name.lower() == "cifar10":
            return self._load_cifar10()
        elif self.dataset_name.lower() == "mnist":
            return self._load_mnist()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
    def _load_cifar10(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load CIFAR-10 dataset."""
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.input_shape = (32, 32, 3)
        
        return x_train, y_train, x_test, y_test
    
    def _load_mnist(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load MNIST dataset."""
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Add channel dimension for CNN
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        
        self.class_names = [str(i) for i in range(10)]
        self.input_shape = (28, 28, 1)
        
        return x_train, y_train, x_test, y_test
    
    def normalize_data(self, x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            x_train: Training images
            x_test: Test images
            
        Returns:
            Normalized training and test images
        """
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        return x_train, x_test
    
    def prepare_labels(self, y_train: np.ndarray, y_test: np.ndarray, num_classes: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert labels to categorical format.
        
        Args:
            y_train: Training labels
            y_test: Test labels
            num_classes: Number of classes (auto-detected if None)
            
        Returns:
            One-hot encoded labels
        """
        if num_classes is None:
            num_classes = len(self.class_names)
        
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        
        return y_train, y_test
    
    def create_validation_split(self, x_train: np.ndarray, y_train: np.ndarray, 
                               validation_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Create validation split from training data.
        
        Args:
            x_train: Training images
            y_train: Training labels
            validation_size: Fraction of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Split training and validation data
        """
        return train_test_split(
            x_train, y_train, 
            test_size=validation_size, 
            random_state=random_state,
            stratify=y_train
        )
    
    def get_data_info(self, x_train: np.ndarray, y_train: np.ndarray, 
                     x_val: np.ndarray, y_val: np.ndarray,
                     x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            'dataset_name': self.dataset_name,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_shape': self.input_shape,
            'train_samples': len(x_train),
            'val_samples': len(x_val),
            'test_samples': len(x_test),
            'train_shape': x_train.shape,
            'pixel_range': f"[{x_train.min():.2f}, {x_train.max():.2f}]"
        }
    
    def visualize_samples(self, x_data: np.ndarray, y_data: np.ndarray, 
                         num_samples: int = 20, title: str = "Sample Images") -> None:
        """
        Visualize sample images from the dataset.
        
        Args:
            x_data: Image data
            y_data: Labels
            num_samples: Number of samples to display
            title: Plot title
        """
        plt.figure(figsize=(15, 8))
        for i in range(min(num_samples, len(x_data))):
            plt.subplot(4, 5, i + 1)
            
            if len(x_data.shape) == 4 and x_data.shape[-1] == 1:
                # Grayscale image
                plt.imshow(x_data[i].squeeze(), cmap='gray')
            else:
                # Color image
                plt.imshow(x_data[i])
            
            # Handle both categorical and integer labels
            if len(y_data.shape) > 1:
                label_idx = np.argmax(y_data[i])
            else:
                label_idx = int(y_data[i])
                
            plt.title(f'{self.class_names[label_idx]}')
            plt.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def create_data_generator(self, rotation_range: int = 20, 
                             width_shift_range: float = 0.2,
                             height_shift_range: float = 0.2,
                             horizontal_flip: bool = True,
                             zoom_range: float = 0.2,
                             shear_range: float = 0.2) -> tf.keras.preprocessing.image.ImageDataGenerator:
        """
        Create data generator for image augmentation.
        
        Args:
            rotation_range: Range of rotation in degrees
            width_shift_range: Range of horizontal shifts
            height_shift_range: Range of vertical shifts
            horizontal_flip: Whether to apply horizontal flips
            zoom_range: Range of zoom
            shear_range: Range of shear transformation
            
        Returns:
            Configured ImageDataGenerator
        """
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            zoom_range=zoom_range,
            shear_range=shear_range,
            fill_mode='nearest'
        )
    
    def visualize_augmentation(self, x_data: np.ndarray, y_data: np.ndarray, 
                              generator: tf.keras.preprocessing.image.ImageDataGenerator,
                              num_images: int = 5) -> None:
        """
        Visualize original vs augmented images.
        
        Args:
            x_data: Original images
            y_data: Labels
            generator: Data generator for augmentation
            num_images: Number of images to show
        """
        fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
        
        for i in range(num_images):
            # Original image
            axes[0, i].imshow(x_data[i])
            
            # Handle both categorical and integer labels
            if len(y_data.shape) > 1:
                label_idx = np.argmax(y_data[i])
            else:
                label_idx = int(y_data[i])
                
            axes[0, i].set_title(f'Original: {self.class_names[label_idx]}')
            axes[0, i].axis('off')
            
            # Augmented image
            aug_image = generator.random_transform(x_data[i])
            axes[1, i].imshow(aug_image)
            axes[1, i].set_title('Augmented')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


def load_custom_dataset(data_dir: str, img_size: Tuple[int, int] = (32, 32)) -> Tuple:
    """
    Load custom dataset from directory structure.
    Expected structure: data_dir/class_name/image_files
    
    Args:
        data_dir: Path to dataset directory
        img_size: Target image size
        
    Returns:
        Tuple of (images, labels, class_names)
    """
    images = []
    labels = []
    class_names = []
    
    # Get class names from directory structure
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
    
    # Load images
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(class_idx)
    
    return np.array(images), np.array(labels), class_names


if _name_ == "_main_":
    # Example usage
    preprocessor = DataPreprocessor("cifar10")
    
    # Load data
    x_train, y_train, x_test, y_test = preprocessor.load_dataset()
    
    # Normalize
    x_train, x_test = preprocessor.normalize_data(x_train, x_test)
    
    # Prepare labels
    y_train, y_test = preprocessor.prepare_labels(y_train, y_test)
    
    # Create validation split
    x_train, x_val, y_train, y_val = preprocessor.create_validation_split(x_train, y_train)
    
    # Get dataset info
    info = preprocessor.get_data_info(x_train, y_train, x_val, y_val, x_test, y_test)
    print("Dataset Information:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Visualize samples
    preprocessor.visualize_samples(x_train, y_train)
