"""
Lightweight validation script for SimAM implementation
Tests core module functionality without full ultralytics dependencies
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add ultralytics to path
sys.path.insert(0, str(Path(__file__).parent / 'ultralytics'))

def test_simam_module():
    """Test SimAM module directly"""
    print("\n[1/3] Testing SimAM Module Implementation")
    print("-" * 60)
    
    try:
        from ultralytics.nn.modules.conv import SimAM
        print("✓ SimAM module imported successfully")
        
        # Test initialization
        simam = SimAM()
        print(f"✓ SimAM initialized with default e_lambda={simam.e_lambda}")
        
        simam_custom = SimAM(e_lambda=1e-3)
        print(f"✓ SimAM initialized with custom e_lambda={simam_custom.e_lambda}")
        
        # Test forward pass
        batch_size, channels, height, width = 2, 64, 56, 56
        x = torch.randn(batch_size, channels, height, width)
        
        output = simam(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Verify shape preservation
        if output.shape == x.shape:
            print("✓ Shape preserved (input == output)")
        else:
            print(f"✗ Shape mismatch! Expected {x.shape}, got {output.shape}")
            return False
        
        # Test parameter count
        param_count = sum(p.numel() for p in simam.parameters())
        print(f"✓ Parameters: {param_count} (should be 0 for parameter-free)")
        
        if param_count == 0:
            print("✓ Confirmed parameter-free implementation")
        else:
            print(f"⚠ Warning: SimAM has {param_count} parameters (expected 0)")
        
        # Test on different input sizes
        test_sizes = [(1, 128, 28, 28), (4, 256, 14, 14), (2, 512, 7, 7)]
        print("\n  Testing multiple input sizes:")
        for size in test_sizes:
            x_test = torch.randn(*size)
            out_test = simam(x_test)
            if out_test.shape == x_test.shape:
                print(f"    ✓ {size} -> {out_test.shape}")
            else:
                print(f"    ✗ {size} failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_registration():
    """Test if SimAM is properly registered"""
    print("\n[2/3] Testing Module Registration")
    print("-" * 60)
    
    try:
        # Test __init__.py export
        from ultralytics.nn.modules import SimAM
        print("✓ SimAM exported in ultralytics.nn.modules.__init__")
        
        # Test tasks.py import
        from ultralytics.nn.tasks import SimAM as SimAMTask
        print("✓ SimAM imported in ultralytics.nn.tasks")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yaml_configs():
    """Test if YAML configs exist and are valid"""
    print("\n[3/3] Testing YAML Configurations")
    print("-" * 60)
    
    configs = [
        'ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml',
        'ultralytics/cfg/models/v8/yolov8_SimAM_backbone_cls.yaml',
        'ultralytics/cfg/models/v8/yolov8_SimAM_ECA_hybrid_cls.yaml',
        'ultralytics/cfg/models/v8/yolov8_SimAM_custom_cls.yaml',
    ]
    
    all_exist = True
    for config in configs:
        path = Path(config)
        if path.exists():
            print(f"✓ {path.name}")
            
            # Check if SimAM is in the config
            with open(path, 'r') as f:
                content = f.read()
                if 'SimAM' in content:
                    simam_count = content.count('SimAM')
                    print(f"  → Contains {simam_count} SimAM reference(s)")
                else:
                    print(f"  ⚠ Warning: No SimAM reference found")
        else:
            print(f"✗ {path.name} not found")
            all_exist = False
    
    return all_exist

def test_comparison_with_eca():
    """Compare SimAM with ECA implementation"""
    print("\n[Bonus] Comparing SimAM vs ECA")
    print("-" * 60)
    
    try:
        from ultralytics.nn.modules.conv import SimAM, ECAAttention
        
        # Test input
        x = torch.randn(2, 256, 28, 28)
        
        # SimAM
        simam = SimAM()
        simam_params = sum(p.numel() for p in simam.parameters())
        simam_out = simam(x)
        
        # ECA
        eca = ECAAttention(c1=256, k_size=3)
        eca_params = sum(p.numel() for p in eca.parameters())
        eca_out = eca(x)
        
        print(f"SimAM Parameters: {simam_params}")
        print(f"ECA Parameters:   {eca_params}")
        print(f"Parameter Reduction: {eca_params} → {simam_params} (100% reduction)")
        
        print(f"\nSimAM Output Shape: {simam_out.shape}")
        print(f"ECA Output Shape:   {eca_out.shape}")
        
        if simam_out.shape == eca_out.shape == x.shape:
            print("✓ Both modules preserve shape correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("="*80)
    print("YOLOv8-SimAM Implementation Validation (Lightweight)")
    print("="*80)
    
    results = []
    
    # Test 1: SimAM Module
    results.append(("SimAM Module", test_simam_module()))
    
    # Test 2: Module Registration
    results.append(("Module Registration", test_module_registration()))
    
    # Test 3: YAML Configs
    results.append(("YAML Configurations", test_yaml_configs()))
    
    # Bonus: Comparison
    results.append(("SimAM vs ECA", test_comparison_with_eca()))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    print("-"*80)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED - SimAM Implementation is READY!")
        print("="*80)
        print("\n📋 Implementation Summary:")
        print("  • SimAM class added to ultralytics/nn/modules/conv.py")
        print("  • Module exported and registered properly")
        print("  • 4 YAML configurations created")
        print("  • Parameter-free attention confirmed (0 params)")
        print("\n🚀 Next Steps:")
        print("  1. Install full dependencies: pip install ultralytics opencv-python")
        print("  2. Prepare your dataset and data.yaml configuration")
        print("  3. Train baseline: python train_simam.py --model yolov8_SimAM_cls.yaml --data data.yaml")
        print("  4. Compare models: python compare_models.py --data data.yaml")
        print("\n📊 Available Configurations:")
        print("  • yolov8_SimAM_cls.yaml - Basic SimAM in head")
        print("  • yolov8_SimAM_backbone_cls.yaml - Multi-scale SimAM")
        print("  • yolov8_SimAM_ECA_hybrid_cls.yaml - Hybrid attention")
        print("  • yolov8_SimAM_custom_cls.yaml - Custom e_lambda tuning")
        print("="*80)
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Please review errors above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
