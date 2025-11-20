"""
Test script to validate torch compile integration.
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sd-scripts'))

def test_compile_utils_import():
    """Test that compile_utils can be imported"""
    print("Test 1: Importing compile_utils...")
    try:
        from library import compile_utils
        print("  [OK] compile_utils imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Failed to import compile_utils: {e}")
        return False


def test_compile_arguments():
    """Test that compile arguments can be added to parser"""
    print("\nTest 2: Adding compile arguments to parser...")
    try:
        from library import compile_utils
        parser = argparse.ArgumentParser()
        compile_utils.add_compile_arguments(parser)
        
        # Parse with compile flags
        args = parser.parse_args([
            '--compile',
            '--compile_backend', 'inductor',
            '--compile_mode', 'reduce-overhead',
            '--compile_dynamic', 'auto',
            '--compile_fullgraph',
            '--compile_cache_size_limit', '32'
        ])
        
        assert args.compile == True, "compile flag not set"
        assert args.compile_backend == 'inductor', "compile_backend not set"
        assert args.compile_mode == 'reduce-overhead', "compile_mode not set"
        assert args.compile_dynamic == 'auto', "compile_dynamic not set"
        assert args.compile_fullgraph == True, "compile_fullgraph not set"
        assert args.compile_cache_size_limit == 32, "compile_cache_size_limit not set"
        
        print("  [OK] All compile arguments parsed correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_state_dict_uncompile():
    """Test state dict uncompiling"""
    print("\nTest 3: Testing state dict uncompiling...")
    try:
        from library import compile_utils
        
        # Test with compiled state dict
        compiled_state = {
            "layer1._orig_mod.weight": "tensor1",
            "layer2._orig_mod.bias": "tensor2",
            "layer3.weight": "tensor3"
        }
        
        uncompiled_state = compile_utils.maybe_uncompile_state_dict(compiled_state)
        
        assert "layer1.weight" in uncompiled_state, "Failed to strip _orig_mod from layer1"
        assert "layer2.bias" in uncompiled_state, "Failed to strip _orig_mod from layer2"
        assert "layer3.weight" in uncompiled_state, "layer3 should remain unchanged"
        assert "_orig_mod" not in str(uncompiled_state.keys()), "Still contains _orig_mod"
        
        print("  [OK] State dict uncompiling works correctly")
        
        # Test with non-compiled state dict
        normal_state = {"layer1.weight": "tensor1", "layer2.bias": "tensor2"}
        result = compile_utils.maybe_uncompile_state_dict(normal_state)
        assert result == normal_state, "Normal state dict should be unchanged"
        
        print("  [OK] Normal state dict passes through correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sdxl_train_import():
    """Test that SDXL training scripts import correctly"""
    print("\nTest 4: Testing SDXL script imports...")
    try:
        # We can't fully import these because they need CUDA, but we can check syntax
        import ast
        
        sdxl_train_path = os.path.join(os.path.dirname(__file__), '..', 'sd-scripts', 'sdxl_train.py')
        sdxl_train_network_path = os.path.join(os.path.dirname(__file__), '..', 'sd-scripts', 'sdxl_train_network.py')
        
        with open(sdxl_train_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print("  [OK] sdxl_train.py syntax valid")
        
        with open(sdxl_train_network_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print("  [OK] sdxl_train_network.py syntax valid")
        
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flux_train_import():
    """Test that FLUX training scripts import correctly"""
    print("\nTest 5: Testing FLUX script imports...")
    try:
        import ast
        
        flux_train_path = os.path.join(os.path.dirname(__file__), '..', 'sd-scripts', 'flux_train.py')
        flux_train_network_path = os.path.join(os.path.dirname(__file__), '..', 'sd-scripts', 'flux_train_network.py')
        
        with open(flux_train_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print("  [OK] flux_train.py syntax valid")
        
        with open(flux_train_network_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print("  [OK] flux_train_network.py syntax valid")
        
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gui_syntax():
    """Test that GUI files have valid syntax"""
    print("\nTest 6: Testing GUI file syntax...")
    try:
        import ast
        
        gui_files = [
            'kohya_gui/class_advanced_training.py',
            'kohya_gui/lora_gui.py',
            'kohya_gui/dreambooth_gui.py',
            'kohya_gui/finetune_gui.py'
        ]
        
        for gui_file in gui_files:
            filepath = os.path.join(os.path.dirname(__file__), '..', gui_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            print(f"  [OK] {gui_file} syntax valid")
        
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("TORCH COMPILE INTEGRATION VALIDATION")
    print("="*70)
    
    tests = [
        test_compile_utils_import,
        test_compile_arguments,
        test_state_dict_uncompile,
        test_sdxl_train_import,
        test_flux_train_import,
        test_gui_syntax,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  [FAIL] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

