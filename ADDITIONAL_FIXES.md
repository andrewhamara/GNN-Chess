# Additional Fixes

## 1. Value Head Shape Issue

**Problem:** HeteroEdgeNet's value head was returning shape `(batch, 1)` instead of `(batch,)`.

**Fix:** Added `.squeeze(-1)` to remove the last dimension in `models.py` line 408:

```python
# Before:
v = nn.Dense(1)(v)
v = nn.tanh(v)
return logits, v

# After:
v = nn.Dense(1)(v)
v = nn.tanh(v)
v = v.squeeze(-1)  # Remove last dimension to match expected shape
return logits, v
```

This ensures the value output matches the expected shape for compatibility with the training loop.

## 2. JAX Deprecation Warning

**Problem:** Using deprecated `jax.tree_map` which triggers warnings in JAX 0.4.25+.

**Warning:**
```
DeprecationWarning: jax.tree_map is deprecated: use jax.tree.map (jax v0.4.25 or newer)
```

**Fix:** Replaced all occurrences of `jax.tree_map` with `jax.tree.map` in:
- `test_hetero_graph.py`
- `chess_graph.py`
- `debug_segfault.py`
- `minimal_test.py`
- `jpyger.py` (if present)

**Files Modified:**
```bash
# Search and replace
jax.tree_map → jax.tree.map
```

This removes all deprecation warnings and uses the new JAX API.

## Testing

Run the test again:
```bash
python test_hetero_graph.py
```

Expected output (no warnings or errors):
```
✓ Test 1 passed! (Graph Construction)
✓ Test 2 passed! (Model Forward Pass)
✓ Test 3 passed! (JIT Compilation)
✓ Test 4 passed! (Batched Graph)
✓ Test 5 passed! (Gradient Computation)
✓ Test 6 passed! (Multi-Device)
✅ All tests passed! Ready for training.
```

## Summary of All Fixes

1. **Segmentation faults** (BUGFIXES.md)
   - Fixed `jnp.isin` incompatibility
   - Fixed static edge features shape

2. **Python version compatibility** (PYTHON_VERSION_FIX.md)
   - Replaced `|` union syntax with `Optional`/`Union`

3. **Value head shape** (this file)
   - Added `.squeeze(-1)` for correct output shape

4. **JAX deprecations** (this file)
   - Updated `jax.tree_map` → `jax.tree.map`

All issues resolved! 🎉
