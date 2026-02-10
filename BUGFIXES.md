# Bug Fixes for Segmentation Fault

## Issues Found and Fixed

### 1. **`jnp.isin` Incompatibility** (chess_graph.py:267)
**Problem:** `jnp.isin()` can cause segfaults or incorrect behavior with JAX arrays, especially in JIT-compiled contexts.

**Fix:** Replaced with explicit boolean comparisons:
```python
# Before:
is_sliding = jnp.isin(safe_piece, _SLIDING_TYPES)

# After:
is_sliding = (
    (safe_piece == 2) | (safe_piece == 3) | (safe_piece == 4) |  # B, R, Q (white)
    (safe_piece == 8) | (safe_piece == 9) | (safe_piece == 10)   # b, r, q (black)
)
```

### 2. **Static Edge Features Shape Mismatch** (chess_graph.py:523)
**Problem:** When tiling static edge features for batching, the reshape was incorrect, causing shape mismatches that could lead to memory corruption.

**Fix:**
```python
# Before:
all_f = jnp.tile(f_j, (batch_size, 1))

# After:
all_f = jnp.tile(f_j, (batch_size, 1)).reshape(-1, f_j.shape[-1])
```

This ensures features have shape `(batch_size * n_edges, n_features)` instead of `(batch_size, n_edges)`.

## Testing the Fixes

Run the following tests in order:

### Test 1: Minimal Graph Construction
```bash
python minimal_test.py
```

Expected output:
```
Importing JAX...
JAX version: 0.4.x
Devices: [...]
Importing PGX...
Importing chess_graph...
Creating environment...
Initializing state...
Observation shape: (1, 8, 8, 119)
Legal action mask shape: (1, 4672)
Constructing HeteroGraph...
SUCCESS!
Nodes: (64, 119)
Move edges: (1858,)
Grid edges: (420,)
Attack edges: (4096,)
Defense edges: (4096,)
All tests passed!
```

### Test 2: Full Verification
```bash
python debug_segfault.py
```

Should pass all 10 tests without segfault.

### Test 3: Complete Test Suite
```bash
python test_hetero_graph.py
```

Should pass all 6 tests.

## Running with Pixi

If you're using the pixi environment:

```bash
# Add pixi to PATH or use full path
pixi run python minimal_test.py
pixi run python debug_segfault.py
pixi run python test_hetero_graph.py

# Or use the pixi task:
pixi run test
```

## If Segfault Persists

If you still see segfaults after these fixes, try:

1. **Check JAX version:**
   ```python
   import jax
   print(jax.__version__)
   # Should be < 0.6 (as specified in pixi.toml)
   ```

2. **Disable JIT temporarily:**
   Add this at the top of `minimal_test.py`:
   ```python
   import jax
   jax.config.update('jax_disable_jit', True)
   ```

3. **Enable NaN/Inf checking:**
   ```python
   jax.config.update('jax_debug_nans', True)
   jax.config.update('jax_debug_infs', True)
   ```

4. **Check for array dtype mismatches:**
   The debug_segfault.py script will show where the issue occurs.

5. **Try CPU-only mode:**
   ```bash
   JAX_PLATFORMS=cpu python minimal_test.py
   ```

   If this works but GPU mode fails, the issue is in JAX's GPU kernels.

## Common Causes of Remaining Segfaults

1. **JAX/CUDA version mismatch** - Reinstall JAX for your CUDA version
2. **Out-of-bounds array access** - Our fixes should have addressed this
3. **Memory corruption from incorrect shapes** - Fixed with the reshape fix
4. **PGX version incompatibility** - Try `pip install --upgrade pgx==2.3.0`

## Known Working Environment

From `pixi.toml`:
```toml
python = "==3.11.9"
jax = "<0.6"
pgx = "<2.4"
jraph = ">=0.0.6.dev0,<0.0.7"
flax = ">=0.10.4,<0.11"
```

## Reporting Issues

If segfault persists, please provide:
1. Output from `debug_segfault.py` (which test fails)
2. JAX version: `python -c "import jax; print(jax.__version__)"`
3. OS and GPU info
4. Full error message / backtrace if available
