#!/usr/bin/env python3
"""Debug script to isolate segmentation fault in hetero graph construction."""

import sys
import traceback

print("=" * 70)
print("Debugging Segmentation Fault")
print("=" * 70)

# Test 1: Basic imports
print("\n[1/8] Testing basic imports...")
try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    print("✓ JAX imports successful")
    print(f"  JAX version: {jax.__version__}")
    print(f"  Devices: {jax.devices()}")
    print(f"  Backend: {jax.default_backend()}")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: PGX import
print("\n[2/8] Testing PGX import...")
try:
    import pgx
    print(f"✓ PGX import successful (version: {pgx.__version__})")
except Exception as e:
    print(f"✗ PGX import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Chess graph import
print("\n[3/8] Testing chess_graph import...")
try:
    import chess_graph as cg
    print("✓ chess_graph import successful")
except Exception as e:
    print(f"✗ chess_graph import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create simple chess state
print("\n[4/8] Creating chess environment...")
try:
    env = pgx.make("chess")
    print("✓ Chess environment created")
except Exception as e:
    print(f"✗ Environment creation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Initialize state
print("\n[5/8] Initializing chess state...")
try:
    state = env.init(jax.random.PRNGKey(0))
    print(f"✓ State initialized")
    print(f"  Observation shape: {state.observation.shape}")
    print(f"  Legal action mask shape: {state.legal_action_mask.shape}")
except Exception as e:
    print(f"✗ State initialization failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Add batch dimension
print("\n[6/8] Adding batch dimension...")
try:
    state = jax.tree_map(lambda x: x[None], state)
    print(f"✓ Batch dimension added")
    print(f"  Observation shape: {state.observation.shape}")
except Exception as e:
    print(f"✗ Batch dimension failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test static edge construction (safe)
print("\n[7/8] Testing static edge construction...")
try:
    static_edges = cg._get_static_edges(8)
    print("✓ Static edges constructed")
    print(f"  Grid edges: {static_edges['grid'][0].shape[0]} edges")
    print(f"  File edges: {static_edges['file'][0].shape[0]} edges")
    print(f"  Rank edges: {static_edges['rank'][0].shape[0]} edges")
    print(f"  Diagonal edges: {static_edges['diagonal'][0].shape[0]} edges")
except Exception as e:
    print(f"✗ Static edges failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 8: Vision lookup tables (potential issue)
print("\n[8/8] Testing vision lookup tables...")
try:
    vision_tables = cg._get_vision_tables(8)
    print("✓ Vision tables constructed")
    print(f"  Can reach shape: {vision_tables['can_reach'].shape}")
    print(f"  Between shape: {vision_tables['between'].shape}")
except Exception as e:
    print(f"✗ Vision tables failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 9: Call _state_nodes
print("\n[9/9] Testing _state_nodes...")
try:
    n_nodes, node_features = cg._state_nodes(state.observation[0])
    print(f"✓ _state_nodes successful")
    print(f"  n_nodes: {n_nodes}")
    print(f"  node_features shape: {node_features.shape}")
except Exception as e:
    print(f"✗ _state_nodes failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 10: Graph construction (THIS IS LIKELY WHERE SEGFAULT HAPPENS)
print("\n[10/10] Testing full graph construction...")
print("NOTE: If segfault occurs, it's likely in state_to_hetero_graph")
try:
    graph = cg.state_to_hetero_graph(state.observation, state.legal_action_mask)
    print("✓ Graph construction successful!")
    print(f"  Nodes shape: {graph.nodes.shape}")
    print(f"  Move edges: {graph.move_edges.senders.shape[0]} edges")
    print(f"  Attack edges: {graph.attack_edges.senders.shape[0]} edges")
    print(f"  Defense edges: {graph.defense_edges.senders.shape[0]} edges")
except Exception as e:
    print(f"✗ Graph construction failed: {e}")
    traceback.print_exc()
    print("\nDEBUG INFO:")
    print(f"  observation shape: {state.observation.shape}")
    print(f"  legal_action_mask shape: {state.legal_action_mask.shape}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ All tests passed! No segfault detected.")
print("=" * 70)
print("\nIf you still see segfault in test_hetero_graph.py, it's likely in:")
print("  1. Model initialization")
print("  2. JIT compilation")
print("  3. Gradient computation")
