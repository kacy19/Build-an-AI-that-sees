"""
Model architectures for Vision AI project.
Contains custom CNN and transfer learning implementations.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from typing import Tuple, Optional
import numpy as np


class ModelBuilder:
    """Class for building different CNN architectures."""
    
    def _init_(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize model builder.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def create_basic_cnn(self) -> tf.keras.Model:
        """
        Create a basic CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            
            # Classification Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_improved_cnn(self) -> tf.keras.Model:
        """
        Create an improved CNN with more layers and regularization.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Classification Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_mobilenet_transfer(self, trainable: bool = False) -> tf.keras.Model:
        """
        Create a transfer learning model using MobileNetV2.
        
        Args:
            trainable: Whether to make base model trainable
            
        Returns:
            Transfer learning model
        """
        # Load pre-trained MobileNetV2
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze/unfreeze base model
        base_model.trainable = trainable
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_resnet_transfer(self, trainable: bool = False) -> tf.keras.Model:
        """
        Create a transfer learning model using ResNet50.
        
        Args:
            trainable: Whether to make base model trainable
            
        Returns:
            Transfer learning model
        """
        # Load pre-trained ResNet50
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze/unfreeze base model
        base_model.trainable = trainable
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_vgg_transfer(self, trainable: bool = False) -> tf.keras.Model:
        """
        Create a transfer learning model using VGG16.
        
        Args:
            trainable: Whether to make base model trainable
            
        Returns:
            Transfer learning model
        """
        # Load pre-trained VGG16
        base_model = applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze/unfreeze base model
        base_model.trainable = trainable
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model: tf.keras.Model, 
                     learning_rate: float = 0.001,
                     optimizer: str = 'adam') -> tf.keras.Model:
        """
        Compile model with specified parameters.
        
        Args:
            model: Model to compile
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            
        Returns:
            Compiled model
        """
        if optimizer.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def get_model_summary(self, model: tf.keras.Model) -> dict:
        """
        Get comprehensive model information.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with model information
        """
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'num_layers': len(model.layers),
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Approximate size in MB
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }


class CustomCNNBlock:
    """Helper class for creating custom CNN blocks."""
    
    @staticmethod
    def conv_block(filters: int, kernel_size: Tuple[int, int] = (3, 3),
                   strides: Tuple[int, int] = (1, 1),
                   padding: str = 'same',
                   use_batch_norm: bool = True,
                   dropout_rate: float = 0.0,
                   activation: str = 'relu') -> list:
        """
        Create a convolutional block.
        
        Args:
            filters: Number of filters
            kernel_size: Kernel size
            strides: Stride values
            padding: Padding type
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0 for no dropout)
            activation: Activation function
            
        Returns:
            List of layers
        """
        block = [
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)
        ]
        
        if use_batch_norm:
            block.append(layers.BatchNormalization())
        
        if dropout_rate > 0:
            block.append(layers.Dropout(dropout_rate))
        
        return block
    
    @staticmethod
    def residual_block(filters: int, kernel_size: Tuple[int, int] = (3, 3)) -> tf.keras.Model:
        """
        Create a residual block.
        
        Args:
            filters: Number of filters
            kernel_size: Kernel size
            
        Returns:
            Residual block as functional model
        """
        def residual_function(x):
            # Main path
            y = layers.Conv2D(filters, kernel_size, padding='same')(x)
            y = layers.BatchNormalization()(y)
            y = layers.Activation('relu')(y)
            
            y = layers.Conv2D(filters, kernel_size, padding='same')(y)
            y = layers.BatchNormalization()(y)
            
            # Skip connection
            if x.shape[-1] != filters:
                x = layers.Conv2D(filters, (1, 1), padding='same')(x)
                x = layers.BatchNormalization()(x)
            
            # Add skip connection
            y = layers.Add()([x, y])
            y = layers.Activation('relu')(y)
            
            return y
        
        return residual_function


def create_custom_architecture(input_shape: Tuple[int, int, int], 
                             num_classes: int,
                             architecture_type: str = 'basic') -> tf.keras.Model:
    """
    Create custom architecture based on type.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        architecture_type: Type of architecture ('basic', 'improved', 'residual')
        
    Returns:
        Custom CNN model
    """
    builder = ModelBuilder(input_shape, num_classes)
    
    if architecture_type == 'basic':
        return builder.create_basic_cnn()
    elif architecture_type == 'improved':
        return builder.create_improved_cnn()
    elif architecture_type == 'residual':
        return create_residual_cnn(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")


def create_residual_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Create a simple ResNet-style CNN.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        ResNet-style model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks
    residual_func = CustomCNNBlock.residual_block(32)
    x = residual_func(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    residual_func = CustomCNNBlock.residual_block(64)
    x = residual_func(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    residual_func = CustomCNNBlock.residual_block(128)
    x = residual_func(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name='residual_cnn')


def get_available_models() -> dict:
    """
    Get dictionary of available model architectures.
    
    Returns:
        Dictionary with model names and descriptions
    """
    return {
        'basic_cnn': 'Basic CNN with 3 convolutional blocks',
        'improved_cnn': 'Improved CNN with batch normalization and dropout',
        'residual_cnn': 'Custom ResNet-style architecture',
        'mobilenet_transfer': 'MobileNetV2 transfer learning',
        'resnet_transfer': 'ResNet50 transfer learning',
        'vgg_transfer': 'VGG16 transfer learning'
    }


if _name_ == "_main_":
    # Example usage
    input_shape = (32, 32, 3)
    num_classes = 10
    
    # Create model builder
    builder = ModelBuilder(input_shape, num_classes)
    
    # Test different architectures
    models_to_test = [
        ('basic_cnn', builder.create_basic_cnn()),
        ('improved_cnn', builder.create_improved_cnn()),
        ('mobilenet_transfer', builder.create_mobilenet_transfer()),
    ]
    
    print("Model Architecture Comparison:")
    print("-" * 80)
    
    for name, model in models_to_test:
        model_info = builder.get_model_summary(model)
        print(f"\n{name.upper()}:")
        print(f"  Total Parameters: {model_info['total_parameters']:,}")
        print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
        print(f"  Model Size: {model_info['model_size_mb']:.2f} MB")
        print(f"  Number of Layers: {model_info['num_layers']}")
    
    print("\nAvailable Models:")
    for name, description in get_available_models().items():
        print(f"  {name}: {description}")
