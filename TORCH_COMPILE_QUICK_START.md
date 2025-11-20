# Torch Compile Quick Start Guide

## 🚀 What is Torch Compile?

PyTorch 2.x's `torch.compile` uses graph compilation to optimize your models for faster training. This implementation adds compile support to SDXL and FLUX training with minimal configuration needed.

**Expected Results**: 10-40% speedup after first epoch (one-time 30-120s compilation overhead)

## ⚡ Quick Start

### Option 1: GUI (Easiest)

1. **Open GUI** (LoRA/DreamBooth/Fine-tune tab)
2. **Select Model**: Check SDXL or FLUX checkbox
3. **Go to Advanced Training** section
4. **Expand "Torch Compile Settings"**
5. **Check "Enable torch.compile"**
6. **Leave other settings as default** (or customize below)
7. **Train normally!**

### Option 2: CLI (For Scripters)

**SDXL LoRA**:
```bash
accelerate launch sdxl_train_network.py \
  --compile \
  --pretrained_model_name_or_path your_model.safetensors \
  --train_data_dir ./dataset \
  --output_dir ./output \
  --network_module networks.lora \
  --network_dim 32 --network_alpha 16 \
  --max_train_steps 1000 \
  --learning_rate 1e-4 \
  # ... other args ...
```

**FLUX LoRA**:
```bash
accelerate launch flux_train_network.py \
  --compile \
  --pretrained_model_name_or_path flux_model.safetensors \
  --clip_l clip_l.safetensors \
  --t5xxl t5xxl.safetensors \
  --ae ae.safetensors \
  --network_module networks.lora_flux \
  --max_train_steps 1000 \
  # ... other args ...
```

### Option 3: Config File (Recommended for Production)

**your_config.toml**:
```toml
# Enable compilation
compile = true
compile_backend = "inductor"
compile_mode = "reduce-overhead"

# Model
pretrained_model_name_or_path = "model.safetensors"

# Dataset
train_data_dir = "./dataset"
output_dir = "./output"

# Training
max_train_steps = 1000
learning_rate = 1e-4
# ... rest of config ...
```

Run with:
```bash
accelerate launch sdxl_train_network.py --config_file your_config.toml
```

## 🎛️ Settings Explained

### Enable torch.compile
- **What**: Master switch for compilation
- **Default**: `False` (off)
- **Recommendation**: Enable for training runs >50 steps

### Compile Backend
- **What**: Which compiler backend to use
- **Choices**: 
  - `inductor` ⭐ (default, best compatibility)
  - `cudagraphs` (fastest but strict)
  - `eager` (no compile, debug mode)
  - `aot_eager` (debug mode with tracing)
- **Recommendation**: Stick with `inductor`

### Compile Mode  
- **What**: Optimization vs compilation time tradeoff
- **Choices**:
  - `default` (balanced)
  - `reduce-overhead` ⭐ (faster, recommended)
  - `max-autotune` (slowest compile, fastest train)
  - `max-autotune-no-cudagraphs` (max-autotune without cudagraphs)
- **Recommendation**: 
  - Quick experiments: `default`
  - Normal training: `reduce-overhead`
  - Long runs (100+ epochs): `max-autotune`

### Dynamic Shapes
- **What**: How to handle varying input sizes
- **Choices**:
  - `auto` ⭐ (default, PyTorch decides)
  - `true` (enable dynamic, slower but flexible)
  - `false` (disable, fastest but rigid)
- **Recommendation**: Leave on `auto` unless:
  - Using multi-resolution training → set `true`
  - Single resolution training → set `false` for max speed

### Fullgraph Mode
- **What**: Compile entire forward pass as one graph
- **Default**: `False`
- **Recommendation**: **Keep disabled** (too brittle for complex models)
- **Note**: May fail with LoRA, block swapping, or custom modules

### Cache Size Limit
- **What**: Max number of compiled graph variations
- **Default**: 0 (uses PyTorch default of 8-32)
- **Recommendation**:
  - Single resolution: Leave at 0
  - Multi-resolution: Set to 64
  - Getting "cache limit" warnings: Set to 128

## 🎯 Common Scenarios

### Scenario 1: First Time User
```toml
compile = true
# That's it! Use defaults
```

### Scenario 2: Faster Training Wanted
```toml
compile = true
compile_mode = "reduce-overhead"  # Quick compile, good speedup
```

### Scenario 3: Maximum Speed (Long Run)
```toml
compile = true  
compile_mode = "max-autotune"      # Best speed
compile_cache_size_limit = 64      # Prevent recompilations
```

### Scenario 4: Multi-Resolution Dataset
```toml
compile = true
compile_dynamic = "true"            # Handle varying sizes
compile_cache_size_limit = 128     # More cache for variations
```

### Scenario 5: FLUX with Block Swapping
```toml
compile = true
blocks_to_swap = 10                # Memory optimization
# Linear layers auto-disabled, still get speedup!
```

## ⚠️ Troubleshooting

### "Compilation failed" Error
**Try**:
1. Set `compile_mode = "default"`
2. Set `compile_dynamic = "false"`
3. Disable `compile_fullgraph`
4. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - Need 2.1.0 or higher

### "Triton not found" Error
**Fix**:
```bash
pip install triton
```
(Required for CUDA compilation)

### Very Slow First Epoch
**Expected**: Compilation takes 30-120s on first epoch
**Check**: Do subsequent epochs run faster? If yes, working correctly!
**Speed up**: Use `compile_mode = "default"` (faster compilation)

### Recompiling Every Batch
**Symptom**: Logs show "Compiling function..." every batch
**Cause**: Dynamic shapes causing cache misses
**Fix**: Set `compile_dynamic = "false"` for single-resolution training

### Checkpoint Won't Load
**Symptom**: "Unexpected key(s) in state_dict"
**Check**: Should be auto-fixed by `maybe_uncompile_state_dict()`
**Workaround**: If persists, train without compile, report bug

### Block Swapping + Compile Crash
**Symptom**: Error about device placement
**Expected**: Should work (linear layers disabled)
**Fix**: Try disabling one feature to isolate issue

## 💡 Pro Tips

### Tip 1: Test First
Always test with `--max_train_steps 10` before long runs:
```bash
accelerate launch sdxl_train_network.py \
  --compile --config_file your_config.toml \
  --max_train_steps 10
```

### Tip 2: Watch the Logs
Look for this message:
```
Compiling SDXL UNet with torch.compile: backend=inductor, mode=reduce-overhead, dynamic=None, fullgraph=False
```
Confirms compilation is active!

### Tip 3: Time Your Epochs
Compare:
- **Without compile**: Note epoch time
- **With compile**: First epoch slower, then faster
- **Speedup**: (baseline_time - compiled_time) / baseline_time × 100%

### Tip 4: Save Compiled Configs
Once you find settings that work, save the config:
- GUI: Click "Save configuration"
- CLI: Keep your TOML file
- Reuse for future trainings!

### Tip 5: Start Conservative
```toml
# Day 1: Test it works
compile = true

# Day 2: Optimize
compile_mode = "reduce-overhead"

# Day 3: Push limits  
compile_mode = "max-autotune"
compile_cache_size_limit = 64
```

## 📊 Quick Decision Matrix

| Your Situation | Recommended Settings |
|----------------|---------------------|
| First time trying compile | `compile=true` (defaults) |
| Quick experiments (<10 epochs) | `compile=true, mode=default` |
| Normal training (10-100 epochs) | `compile=true, mode=reduce-overhead` |
| Long training (100+ epochs) | `compile=true, mode=max-autotune` |
| Multi-resolution buckets | `compile=true, dynamic=true, cache_limit=64` |
| Low VRAM (using block swap) | `compile=true, blocks_to_swap=X` (auto-compatible) |
| Maximum speed, have time | `compile=true, mode=max-autotune, cache_limit=128` |

## 🎯 When NOT to Use Compile

- ❌ Training <10 steps (overhead not worth it)
- ❌ Debugging model issues (use eager mode)
- ❌ PyTorch <2.1 (not supported)
- ❌ Very custom model modifications (may not compile)
- ❌ Extremely low VRAM (compilation adds ~200-500MB)

## ✅ Checklist for First Use

- [ ] PyTorch version >= 2.1.0
- [ ] Triton installed (for CUDA)
- [ ] Using SDXL or FLUX (not SD1.5/2.x)
- [ ] Normal training run (not debugging)
- [ ] Willing to wait 30-120s on first epoch
- [ ] Ready for 10-40% speedup on subsequent epochs!

## 🆘 Getting Help

If you encounter issues:

1. **Check logs** for "Compiling..." messages
2. **Try default settings** first
3. **Test without compile** to isolate issue
4. **Check PyTorch/Triton versions**
5. **Review TORCH_COMPILE_IMPLEMENTATION_SUMMARY.md** for details

## 🎉 Success Indicators

You'll know it's working when:
- ✅ First epoch takes longer (compilation happening)
- ✅ Logs show "Compiling [model] with torch.compile..."
- ✅ Subsequent epochs are faster
- ✅ Checkpoints save and load correctly
- ✅ No errors or warnings about compilation

Enjoy your faster training! 🚀

