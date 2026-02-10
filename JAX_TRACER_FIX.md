# JAX Tracer Leak Fix

## Issue

**Error:**
```
jax.errors.UnexpectedTracerError: Encountered an unexpected tracer.
A function transformed by JAX had a side effect, allowing for a reference
to an intermediate value with type bool[12,64,64] wrapped in a DynamicJaxprTracer
to escape the scope of the transformation.
```

**Root Cause:** Lazy initialization of lookup tables inside JIT-compiled functions.

When `state_to_hetero_graph` (which is JIT-compiled) called `_get_vision_tables()`, it would:
1. Create new JAX arrays inside the JIT context
2. Store them in a global dictionary
3. This violates JAX's functional programming requirements

## Solution

**Precompute all lookup tables at module import time** (before any JIT compilation).

### Before (broken):
```python
_VISION_TABLES: dict = {}

def _get_vision_tables(size: int):
    if size not in _VISION_TABLES:
        # ❌ Creating JAX arrays inside JIT context
        _VISION_TABLES[size] = {
            'can_reach': jnp.array(_build_piece_can_reach(size)),
            'between': jnp.array(_build_between(size)),
        }
    return _VISION_TABLES[size]
```

### After (fixed):
```python
# ✅ Precompute at module load time (outside JIT)
_VISION_TABLES_8x8 = {
    'can_reach': jnp.array(_build_piece_can_reach(8)),
    'between': jnp.array(_build_between(8)),
}
_VISION_TABLES_5x5 = {
    'can_reach': jnp.array(_build_piece_can_reach(5)),
    'between': jnp.array(_build_between(5)),
}

def _get_vision_tables(size: int):
    """Get precomputed vision tables for the given board size."""
    if size == 8:
        return _VISION_TABLES_8x8
    elif size == 5:
        return _VISION_TABLES_5x5
    else:
        raise ValueError(f"Unsupported board size: {size}. Use 5 or 8.")
```

## Changes Made

1. **Vision tables** (`chess_graph.py` lines 205-221)
   - Precomputed `_VISION_TABLES_8x8` and `_VISION_TABLES_5x5`
   - `_get_vision_tables()` now just returns the precomputed tables

2. **Static edge tables** (`chess_graph.py` lines 109-127)
   - Precomputed `_STATIC_EDGES_8x8` and `_STATIC_EDGES_5x5`
   - `_get_static_edges()` now just returns the precomputed tables

## Why This Works

JAX's JIT compilation requires pure functions with no side effects:
- ✅ Reading global constants is OK
- ❌ Creating/modifying global state inside JIT is NOT OK

By precomputing the tables when the module loads:
- Tables are created in normal Python context (not JIT)
- JIT-compiled functions just read the constants
- No tracer leaks!

## Trade-offs

**Pros:**
- No tracer leaks
- Slightly faster (no lazy initialization overhead)
- More predictable behavior

**Cons:**
- Both board sizes (5×5 and 8×8) are precomputed even if only one is used
- ~10-20 MB extra memory at import time (negligible)

## Verification

Run the test again:
```bash
python test_hetero_graph.py
```

Should now pass without tracer errors!

## Related JAX Concepts

- **Tracers**: JAX's intermediate representations during compilation
- **Pure functions**: Functions with no side effects (required for JIT)
- **Static vs traced values**: Constants vs values that depend on inputs

Read more: https://jax.readthedocs.io/en/latest/errors.html#jax.errors.UnexpectedTracerError
