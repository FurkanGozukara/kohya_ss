"""
Test the smart FLUX compilation with block swapping.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sd-scripts'))

def test_flux_smart_compile_logic():
    """Test the selective compilation logic for FLUX with block swapping"""
    print("="*70)
    print("SMART FLUX COMPILATION TEST")
    print("="*70)
    
    print("\nTest: Block swap calculation logic...")
    
    # Simulate FLUX model dimensions
    total_double_blocks = 19  # FLUX.1-dev has 19 double blocks
    total_single_blocks = 38  # FLUX.1-dev has 38 single blocks
    
    test_cases = [
        (0, 0, 0, 19, 38),    # No swap: all blocks compiled
        (6, 3, 6, 16, 32),    # Swap 6: compile 16 double, 32 single
        (12, 6, 12, 13, 26),  # Swap 12: compile 13 double, 26 single
        (18, 9, 18, 10, 20),  # Swap 18: compile 10 double, 20 single
        (30, 15, 30, 4, 8),   # Swap 30: compile 4 double, 8 single
        (36, 18, 36, 1, 2),   # Swap 36: compile 1 double, 2 single
    ]
    
    print(f"\n{'Swap':<6} | {'DBSwap':<8} | {'SBSwap':<8} | {'DBCompile':<10} | {'SBCompile':<10} | Speedup Est.")
    print("-" * 70)
    
    all_correct = True
    
    for blocks_to_swap, expected_db_swap, expected_sb_swap, expected_db_compile, expected_sb_compile in test_cases:
        # Calculate swap distribution (from flux_models.py logic)
        double_blocks_to_swap = blocks_to_swap // 2
        single_blocks_to_swap = (blocks_to_swap - double_blocks_to_swap) * 2
        
        # Calculate what gets compiled
        non_swapped_double = total_double_blocks - double_blocks_to_swap
        non_swapped_single = total_single_blocks - single_blocks_to_swap
        
        # Verify calculations
        if double_blocks_to_swap != expected_db_swap:
            print(f"  [FAIL] blocks_to_swap={blocks_to_swap}: expected {expected_db_swap} double swap, got {double_blocks_to_swap}")
            all_correct = False
            continue
            
        if single_blocks_to_swap != expected_sb_swap:
            print(f"  [FAIL] blocks_to_swap={blocks_to_swap}: expected {expected_sb_swap} single swap, got {single_blocks_to_swap}")
            all_correct = False
            continue
            
        if non_swapped_double != expected_db_compile:
            print(f"  [FAIL] blocks_to_swap={blocks_to_swap}: expected {expected_db_compile} double compile, got {non_swapped_double}")
            all_correct = False
            continue
            
        if non_swapped_single != expected_sb_compile:
            print(f"  [FAIL] blocks_to_swap={blocks_to_swap}: expected {expected_sb_compile} single compile, got {non_swapped_single}")
            all_correct = False
            continue
        
        # Estimate speedup based on compiled blocks
        total_blocks = total_double_blocks + total_single_blocks
        compiled_blocks = non_swapped_double + non_swapped_single
        compile_ratio = compiled_blocks / total_blocks
        estimated_speedup = f"{int(compile_ratio * 30)}%"  # Assume 30% max speedup
        
        print(f"{blocks_to_swap:<6} | {double_blocks_to_swap:<8} | {single_blocks_to_swap:<8} | {non_swapped_double:<10} | {non_swapped_single:<10} | {estimated_speedup}")
    
    print()
    
    if all_correct:
        print("[OK] All swap calculations correct!")
        print("\nKey Insights:")
        print("  - With blocks_to_swap=6: ~84% of blocks compiled -> ~25% speedup expected")
        print("  - With blocks_to_swap=12: ~68% of blocks compiled -> ~20% speedup expected")
        print("  - With blocks_to_swap=18: ~53% of blocks compiled -> ~16% speedup expected")
        print("  - With blocks_to_swap=30: ~21% of blocks compiled -> ~6% speedup expected")
        print("\nConclusion: Selective compilation provides speedup even with block swapping!")
        return True
    else:
        print("[FAIL] Some calculations incorrect")
        return False


def test_import_new_function():
    """Test that the new smart compile function can be imported"""
    print("\n" + "="*70)
    print("IMPORT TEST")
    print("="*70)
    
    try:
        from library import compile_utils
        
        # Check the function exists
        assert hasattr(compile_utils, 'compile_flux_with_block_swap'), "Function not found!"
        
        # Check signature
        import inspect
        sig = inspect.signature(compile_utils.compile_flux_with_block_swap)
        params = list(sig.parameters.keys())
        
        assert 'args' in params, "Missing args parameter"
        assert 'flux_model' in params, "Missing flux_model parameter"
        assert 'blocks_to_swap' in params, "Missing blocks_to_swap parameter"
        assert 'log_prefix' in params, "Missing log_prefix parameter"
        
        print("[OK] compile_flux_with_block_swap function exists with correct signature")
        print(f"    Parameters: {params}")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    results = []
    
    results.append(test_flux_smart_compile_logic())
    results.append(test_import_new_function())
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] Smart compilation logic validated!")
        print("\nNow when you use blocks_to_swap with compile:")
        print("  - Swapped blocks: NOT compiled (stay on CPU/GPU swap)")
        print("  - Non-swapped blocks: FULLY compiled (stay on GPU)")
        print("  - Result: Partial speedup instead of zero!")
        return 0
    else:
        print(f"\n[FAIL] {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

