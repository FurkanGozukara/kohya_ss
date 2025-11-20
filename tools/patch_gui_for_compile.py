"""
Script to patch GUI files to add torch compile parameters.
This script updates function signatures and UI bindings systematically.
"""

import re
import os

# Compile parameters to add
COMPILE_PARAMS = """    # Torch compile parameters
    compile,
    compile_backend,
    compile_mode,
    compile_dynamic,
    compile_fullgraph,
    compile_cache_size_limit,"""

# Config data entries to add (for train_model functions)
COMPILE_CONFIG_DATA = """        # Torch compile parameters (for SDXL and FLUX)
        "compile": compile if (sdxl or flux1_checkbox) else None,
        "compile_backend": compile_backend if compile and (sdxl or flux1_checkbox) else None,
        "compile_mode": compile_mode if compile and (sdxl or flux1_checkbox) else None,
        "compile_dynamic": compile_dynamic if compile and (sdxl or flux1_checkbox) and compile_dynamic != "auto" else None,
        "compile_fullgraph": compile_fullgraph if compile and (sdxl or flux1_checkbox) else None,
        "compile_cache_size_limit": int(compile_cache_size_limit) if compile and (sdxl or flux1_checkbox) and compile_cache_size_limit > 0 else None,"""

# UI binding entries
COMPILE_UI_BINDINGS = """            # Torch compile parameters
            advanced_training.compile,
            advanced_training.compile_backend,
            advanced_training.compile_mode,
            advanced_training.compile_dynamic,
            advanced_training.compile_fullgraph,
            advanced_training.compile_cache_size_limit,"""


def patch_function_signature(content, function_name, marker_before_insertion):
    """
    Add compile parameters to a function signature before the closing ).
    
    Args:
        content: File content as string
        function_name: Name of function to patch
        marker_before_insertion: Text pattern right before where we insert
    """
    # Find the function
    pattern = rf'(def {function_name}\([^)]*?{re.escape(marker_before_insertion)}[^)]*?)\n(\):)'
    
    def replacer(match):
        params_part = match.group(1)
        closing = match.group(2)
        
        # Check if compile params already added
        if 'compile,' in params_part:
            return match.group(0)  # Already patched
            
        return params_part + ',\n' + COMPILE_PARAMS + '\n' + closing
    
    return re.sub(pattern, replacer, content, flags=re.MULTILINE | re.DOTALL)


def patch_file(filepath, patches):
    """
    Apply patches to a file.
    
    Args:
        filepath: Path to file
        patches: List of (search, replace, description) tuples
    """
    print(f"\nPatching {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"  [!] File not found, skipping")
        return False
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    patches_applied = 0
    
    for search, replace, description in patches:
        if search in content:
            if replace in content:
                print(f"  [OK] {description} - already applied")
            else:
                content = content.replace(search, replace, 1)
                patches_applied += 1
                print(f"  [OK] {description} - applied")
        else:
            print(f"  [WARN] {description} - pattern not found")
    
    if content != original_content and patches_applied > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [DONE] Saved {patches_applied} changes")
        return True
    else:
        print(f"  [INFO] No changes needed")
        return False


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gui_dir = os.path.join(base_dir, 'kohya_gui')
    
    print("="*60)
    print("Torch Compile GUI Integration Patcher")
    print("="*60)
    
    # Patch dreambooth_gui.py
    dreambooth_patches = [
        # Update save_configuration signature
        (
            """    apply_t5_attn_mask,
):
    # Get list of function parameters and values
    parameters = list(locals().items())""",
            """    apply_t5_attn_mask,
    # Torch compile parameters
    compile,
    compile_backend,
    compile_mode,
    compile_dynamic,
    compile_fullgraph,
    compile_cache_size_limit,
):
    # Get list of function parameters and values
    parameters = list(locals().items())""",
            "save_configuration signature"
        ),
        # Update train_model config_toml_data
        (
            """        "blocks_to_swap": blocks_to_swap if flux1_checkbox or sd3_checkbox else None,
    }

    # Given dictionary `config_toml_data`""",
            """        "blocks_to_swap": blocks_to_swap if flux1_checkbox or sd3_checkbox else None,
        # Torch compile parameters (for SDXL and FLUX)
        "compile": compile if (sdxl or flux1_checkbox) else None,
        "compile_backend": compile_backend if compile and (sdxl or flux1_checkbox) else None,
        "compile_mode": compile_mode if compile and (sdxl or flux1_checkbox) else None,
        "compile_dynamic": compile_dynamic if compile and (sdxl or flux1_checkbox) and compile_dynamic != "auto" else None,
        "compile_fullgraph": compile_fullgraph if compile and (sdxl or flux1_checkbox) else None,
        "compile_cache_size_limit": int(compile_cache_size_limit) if compile and (sdxl or flux1_checkbox) and compile_cache_size_limit > 0 else None,
    }

    # Given dictionary `config_toml_data`""",
            "train_model config_toml_data"
        ),
    ]
    
    # Patch finetune_gui.py
    finetune_patches = [
        # Update save_configuration signature
        (
            """    apply_t5_attn_mask,
):
    # Get list of function parameters and values
    parameters = list(locals().items())""",
            """    apply_t5_attn_mask,
    # Torch compile parameters
    compile,
    compile_backend,
    compile_mode,
    compile_dynamic,
    compile_fullgraph,
    compile_cache_size_limit,
):
    # Get list of function parameters and values
    parameters = list(locals().items())""",
            "save_configuration signature"
        ),
        # Update train_model config_toml_data
        (
            """        "blocks_to_swap": blocks_to_swap if flux1_checkbox or sd3_checkbox else None,
    }

    # Given dictionary `config_toml_data`""",
            """        "blocks_to_swap": blocks_to_swap if flux1_checkbox or sd3_checkbox else None,
        # Torch compile parameters (for SDXL and FLUX)
        "compile": compile if (sdxl or flux1_checkbox) else None,
        "compile_backend": compile_backend if compile and (sdxl or flux1_checkbox) else None,
        "compile_mode": compile_mode if compile and (sdxl or flux1_checkbox) else None,
        "compile_dynamic": compile_dynamic if compile and (sdxl or flux1_checkbox) and compile_dynamic != "auto" else None,
        "compile_fullgraph": compile_fullgraph if compile and (sdxl or flux1_checkbox) else None,
        "compile_cache_size_limit": int(compile_cache_size_limit) if compile and (sdxl or flux1_checkbox) and compile_cache_size_limit > 0 else None,
    }

    # Given dictionary `config_toml_data`""",
            "train_model config_toml_data"
        ),
    ]
    
    patch_file(os.path.join(gui_dir, 'dreambooth_gui.py'), dreambooth_patches)
    patch_file(os.path.join(gui_dir, 'finetune_gui.py'), finetune_patches)
    
    print("\n" + "="*60)
    print("Patching complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update settings_list in each GUI to include compile UI components")
    print("2. Test config save/load functionality")
    print("3. Run integration tests")


if __name__ == '__main__':
    main()

