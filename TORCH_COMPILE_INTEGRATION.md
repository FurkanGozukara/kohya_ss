# Torch Compile Integration Progress

## Completed Backend Changes

### ✅ Core Infrastructure
- [x] Created `library/compile_utils.py` with:
  - `add_compile_arguments()` - Adds CLI args to parser
  - `compile_model()` - Compiles model blocks with proper configuration
  - `disable_linear_from_compile()` - Disables compile for Linear layers when needed
  - `maybe_uncompile_state_dict()` - Strips `_orig_mod.` prefixes when saving

### ✅ SDXL Support
- [x] `sdxl_train.py`:
  - Imports `compile_utils`
  - Adds compile arguments via `compile_utils.add_compile_arguments(parser)`
  - Compiles UNet blocks after accelerator.prepare()
  
- [x] `sdxl_train_network.py`:
  - Imports `compile_utils`
  - Overrides `prepare_unet_with_accelerator()` to compile after preparation
  - Adds compile arguments to parser
  
- [x] `library/sdxl_model_util.py`:
  - Updated `save_stable_diffusion_checkpoint()` to strip compiled state dict

### ✅ FLUX Support
- [x] `flux_train.py`:
  - Imports `compile_utils`
  - Compiles FLUX double/single blocks after accelerator.prepare()
  - Handles block swapping compatibility (disables linear layers)
  - Adds compile arguments to parser
  
- [x] `flux_train_network.py`:
  - Imports `compile_utils`
  - Overrides `prepare_unet_with_accelerator()` for both swap and non-swap modes
  - Properly disables linear layers when swapping blocks
  - Adds compile arguments to parser
  
- [x] `library/flux_train_utils.py`:
  - Updated `save_models()` to strip compiled state dict

### ✅ GUI - Advanced Training
- [x] `kohya_gui/class_advanced_training.py`:
  - Added "Torch Compile Settings" accordion with:
    - `compile` checkbox
    - `compile_backend` dropdown (inductor/cudagraphs/eager/aot_eager)
    - `compile_mode` dropdown (default/reduce-overhead/max-autotune/max-autotune-no-cudagraphs)
    - `compile_dynamic` dropdown (auto/true/false)
    - `compile_fullgraph` checkbox
    - `compile_cache_size_limit` number input

## Remaining GUI Integration Work

### 🔄 LoRA GUI (lora_gui.py)
**Status**: Function signatures updated, need to:
1. ✅ Add compile params to `save_configuration()` signature - DONE
2. ✅ Add compile params to `open_configuration()` signature - DONE  
3. ✅ Add compile params to `train_model()` signature - DONE
4. ✅ Update config_toml_data dictionary in train_model - DONE
5. ✅ Add compile params to settings_list for UI binding - DONE
6. ⚠️ Need to verify button_save_config.click() has all parameters

### 🔄 DreamBooth GUI (dreambooth_gui.py)
**Status**: Need to:
1. ⚠️ Add compile params to `save_configuration()` signature (line ~61)
2. ⚠️ Add compile params to `open_configuration()` signature  
3. ⚠️ Add compile params to `train_model()` signature (line ~513)
4. ⚠️ Update config_toml_data dictionary in train_model
5. ⚠️ Add compile params to settings_list for UI binding

### 🔄 Fine-tune GUI (finetune_gui.py)
**Status**: Need to:
1. ⚠️ Add compile params to `save_configuration()` signature (line ~66)
2. ⚠️ Add compile params to `open_configuration()` signature  
3. ⚠️ Add compile params to `train_model()` signature (line ~542)
4. ⚠️ Update config_toml_data dictionary in train_model
5. ⚠️ Add compile params to settings_list for UI binding

## Testing Checklist

### Unit Tests
- [ ] Test `compile_utils.compile_model()` with mock args
- [ ] Test `maybe_uncompile_state_dict()` with compiled and non-compiled dicts

### Integration Tests  
- [ ] SDXL DreamBooth with compile enabled
- [ ] SDXL LoRA with compile enabled
- [ ] FLUX DreamBooth with compile enabled
- [ ] FLUX DreamBooth with compile + block swapping
- [ ] FLUX LoRA with compile enabled
- [ ] FLUX LoRA with compile + block swapping
- [ ] Verify config save/load preserves compile settings
- [ ] Verify checkpoint loading works with uncompiled state dicts

### GUI Tests
- [ ] Open LoRA tab, enable SDXL, verify compile section appears
- [ ] Open LoRA tab, enable FLUX, verify compile section appears
- [ ] Configure compile settings, save config, reload - verify persistence
- [ ] Generate command and verify compile flags appear in CLI

## Known Limitations

1. **CPU Offload Checkpointing**: May have compatibility issues with torch.compile (warning added)
2. **DeepSpeed**: Compile integration not tested with DeepSpeed
3. **Multi-GPU**: Compile should work but not extensively tested
4. **Block Swapping + Compile**: Linear layers are automatically disabled (as per Musubi design)

## Usage Examples

### CLI Example (SDXL LoRA)
```bash
accelerate launch sdxl_train_network.py \
  --compile \
  --compile_backend inductor \
  --compile_mode reduce-overhead \
  --compile_dynamic auto \
  --pretrained_model_name_or_path model.safetensors \
  --network_module networks.lora \
  ...other args...
```

### CLI Example (FLUX with Block Swapping)
```bash
accelerate launch flux_train_network.py \
  --compile \
  --compile_backend inductor \
  --compile_mode default \
  --blocks_to_swap 10 \
  ...other args...
```
(Linear layers automatically disabled when blocks_to_swap > 0)

### TOML Config Example
```toml
[general]
pretrained_model_name_or_path = "model.safetensors"
...

[training]
compile = true
compile_backend = "inductor"
compile_mode = "reduce-overhead"
compile_dynamic = "auto"  # or "true" / "false"
compile_fullgraph = false
compile_cache_size_limit = 32  # optional, 0 = use PyTorch default
...
```

## Reference Implementation

The implementation is based on musubi-tuner commits:
- `fe044b6` - Core compile_transformer helper
- `e1157b3` - State dict handling for compiled models  
- `967a1d4` - Dynamic shapes CLI string handling
- `8e37da6` - Timestep float32 precision fixes

Key design decisions mirrored from Musubi:
1. Compile per-block rather than entire model (better with LoRA/adapters)
2. Disable Linear layers when using memory-saving features (swap/offload)
3. Strip `_orig_mod.` prefixes before saving checkpoints
4. Use string choices for dynamic ("true"/"false"/"auto") mapped to bool/None
5. Keep timesteps as float32 even under mixed precision

