"""
Quick validation script to test SimAM implementation
Tests model loading, forward pass, and parameter counting
"""

import torch
from ultralytics import YOLO
import sys
from pathlib import Path

def test_model_loading(config_path):
    """Test if model config loads successfully"""
    try:
        model = YOLO(config_path)
        return True, model
    except Exception as e:
        return False, str(e)

def test_forward_pass(model, batch_size=2, imgsz=224, num_classes=39):
    """Test forward pass with dummy input"""
    try:
        dummy_input = torch.randn(batch_size, 3, imgsz, imgsz)
        with torch.no_grad():
            output = model.model(dummy_input)
        
        # Verify output shape
        expected_shape = torch.Size([batch_size, num_classes])
        if output.shape == expected_shape:
            return True, f"Output shape: {output.shape} ✓"
        else:
            return False, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    except Exception as e:
        return False, str(e)

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    return total_params, trainable_params

def find_simam_modules(model):
    """Find SimAM modules in the model"""
    simam_count = 0
    simam_locations = []
    
    for i, (name, module) in enumerate(model.model.named_modules()):
        if 'SimAM' in module.__class__.__name__:
            simam_count += 1
            simam_locations.append((i, name, module))
    
    return simam_count, simam_locations

def main():
    print("="*80)
    print("YOLOv8-SimAM Implementation Validation")
    print("="*80)
    
    # Test configurations
    configs = [
        ('yolov8_SimAM_cls.yaml', 'SimAM Basic'),
        ('yolov8_SimAM_backbone_cls.yaml', 'SimAM Backbone'),
        ('yolov8_SimAM_ECA_hybrid_cls.yaml', 'SimAM-ECA Hybrid'),
        ('yolov8_SimAM_custom_cls.yaml', 'SimAM Custom'),
    ]
    
    base_path = Path('ultralytics/cfg/models/v8')
    
    results = []
    
    for config_file, config_name in configs:
        config_path = base_path / config_file
        
        print(f"\n{'-'*80}")
        print(f"Testing: {config_name}")
        print(f"Config: {config_path}")
        print(f"{'-'*80}")
        
        # Test 1: Model Loading
        print("\n[1/4] Testing model loading...")
        success, result = test_model_loading(str(config_path))
        if success:
            print("✓ Model loaded successfully")
            model = result
        else:
            print(f"✗ Failed to load model: {result}")
            results.append({
                'config': config_name,
                'status': 'FAILED',
                'error': result
            })
            continue
        
        # Test 2: Forward Pass
        print("\n[2/4] Testing forward pass...")
        success, message = test_forward_pass(model)
        if success:
            print(f"✓ Forward pass successful: {message}")
        else:
            print(f"✗ Forward pass failed: {message}")
            results.append({
                'config': config_name,
                'status': 'FAILED',
                'error': message
            })
            continue
        
        # Test 3: Parameter Count
        print("\n[3/4] Counting parameters...")
        total_params, trainable_params = count_parameters(model)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Parameters (M): {total_params/1e6:.2f}")
        
        # Test 4: Find SimAM Modules
        print("\n[4/4] Detecting SimAM modules...")
        simam_count, simam_locations = find_simam_modules(model)
        print(f"✓ Found {simam_count} SimAM module(s)")
        
        if simam_count > 0:
            for idx, (i, name, module) in enumerate(simam_locations):
                print(f"  {idx+1}. {module.__class__.__name__} at layer {i}")
                if hasattr(module, 'e_lambda'):
                    print(f"     e_lambda: {module.e_lambda}")
        
        results.append({
            'config': config_name,
            'status': 'PASSED',
            'params': total_params,
            'params_M': round(total_params/1e6, 2),
            'simam_count': simam_count
        })
        
        print(f"\n✓ All tests passed for {config_name}")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    
    print(f"\nTotal configs tested: {len(configs)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if passed > 0:
        print("\nSuccessful Configurations:")
        print(f"{'Config':<30} {'Params (M)':<15} {'SimAM Count':<15}")
        print("-"*60)
        for r in results:
            if r['status'] == 'PASSED':
                print(f"{r['config']:<30} {r['params_M']:<15} {r['simam_count']:<15}")
    
    if failed > 0:
        print("\nFailed Configurations:")
        for r in results:
            if r['status'] == 'FAILED':
                print(f"  - {r['config']}: {r['error']}")
    
    print("\n" + "="*80)
    
    if failed == 0:
        print("✓ ALL TESTS PASSED - Implementation is ready!")
        print("\nNext steps:")
        print("  1. Prepare your dataset and data.yaml")
        print("  2. Run: python train_simam.py --model yolov8_SimAM_cls.yaml --data /path/to/data.yaml --scale n")
        print("  3. Compare models: python compare_models.py --data /path/to/data.yaml")
    else:
        print("✗ SOME TESTS FAILED - Please check the errors above")
        return 1
    
    print("="*80)
    return 0

if __name__ == '__main__':
    sys.exit(main())
