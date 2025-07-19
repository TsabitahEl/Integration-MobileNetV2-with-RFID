#!/usr/bin/env python3
"""
Test script to verify TensorFlow GPU functionality on Jetson Orin
"""

import tensorflow as tf
import sys
import os

def test_tensorflow_gpu():
    print("=" * 60)
    print("TensorFlow GPU Test for Jetson Orin")
    print("=" * 60)
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPUs available: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            
        # Configure GPU memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"❌ GPU memory growth configuration failed: {e}")
    
    # Check CUDA version
    print(f"\nCUDA version: {tf.test.gpu_device_name()}")
    
    # Test basic GPU computation
    print("\nTesting GPU computation...")
    try:
        with tf.device('/GPU:0'):
            # Create some tensors
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print(f"Matrix multiplication result: \n{c}")
            print("✅ GPU computation test passed")
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
    
    # Test TensorFlow model loading capability
    print("\nTesting SavedModel loading capability...")
    try:
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Create dummy data
        import numpy as np
        x_test = np.random.random((10, 5))
        
        # Test prediction
        predictions = model.predict(x_test)
        print(f"Model prediction shape: {predictions.shape}")
        print("✅ Model loading and prediction test passed")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    # Display detailed device info
    print("\nDetailed GPU information:")
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    for device in local_device_protos:
        if device.device_type == 'GPU':
            print(f"Device: {device.name}")
            print(f"Type: {device.device_type}")
            print(f"Memory limit: {device.memory_limit}")
            print(f"Description: {device.physical_device_desc}")
    
    print("\n" + "=" * 60)
    print("GPU Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_tensorflow_gpu()
