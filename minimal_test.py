#!/usr/bin/env python3
"""Minimal test to isolate segfault."""

print("Importing JAX...")
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

print("\nImporting PGX...")
import pgx

print("\nImporting chess_graph...")
import chess_graph as cg

print("\nCreating environment...")
env = pgx.make("chess")

print("Initializing state...")
state = env.init(jax.random.PRNGKey(0))
state = jax.tree.map(lambda x: x[None], state)

print(f"Observation shape: {state.observation.shape}")
print(f"Legal action mask shape: {state.legal_action_mask.shape}")

print("\nConstructing HeteroGraph...")
try:
    graph = cg.state_to_hetero_graph(state.observation, state.legal_action_mask)
    print("SUCCESS!")
    print(f"Nodes: {graph.nodes.shape}")
    print(f"Move edges: {graph.move_edges.senders.shape}")
    print(f"Grid edges: {graph.grid_edges.senders.shape}")
    print(f"Attack edges: {graph.attack_edges.senders.shape}")
    print(f"Defense edges: {graph.defense_edges.senders.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nAll tests passed!")
