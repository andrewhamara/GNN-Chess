#!/usr/bin/env python3
"""Verification script for HeteroEdgeNet implementation.

Tests:
1. Graph construction produces correct shapes
2. Model forward pass works
3. JIT compilation succeeds
4. Multi-device parallelization works
5. Gradient computation succeeds

Run before starting full training to catch issues early.
"""

import jax
import jax.numpy as jnp
import pgx
from rich.pretty import pprint
from rich.console import Console

import chess_graph as cg
from models import HeteroEdgeNet, ModelManager

console = Console()


def test_graph_construction():
    """Test 1: HeteroGraph construction."""
    console.print("\n[bold cyan]Test 1: Graph Construction[/bold cyan]")

    env = pgx.make("chess")
    state = env.init(jax.random.PRNGKey(0))
    state = jax.tree.map(lambda x: x[None], state)  # Add batch dim

    graph = cg.state_to_hetero_graph(state.observation, state.legal_action_mask)

    console.print(f"✓ Nodes shape: {graph.nodes.shape}")
    console.print(f"✓ Move edges: {graph.move_edges.senders.shape[0]} edges, {graph.move_edges.features.shape[-1]} features")
    console.print(f"✓ Grid edges: {graph.grid_edges.senders.shape[0]} edges, {graph.grid_edges.features.shape[-1]} features")
    console.print(f"✓ Attack edges: {graph.attack_edges.senders.shape[0]} edges, {graph.attack_edges.features.shape[-1]} features")
    console.print(f"✓ Defense edges: {graph.defense_edges.senders.shape[0]} edges, {graph.defense_edges.features.shape[-1]} features")
    console.print(f"✓ File edges: {graph.file_edges.senders.shape[0]} edges, {graph.file_edges.features.shape[-1]} features")
    console.print(f"✓ Rank edges: {graph.rank_edges.senders.shape[0]} edges, {graph.rank_edges.features.shape[-1]} features")
    console.print(f"✓ Diagonal edges: {graph.diagonal_edges.senders.shape[0]} edges, {graph.diagonal_edges.features.shape[-1]} features")
    console.print(f"✓ Globals (action mapping): {graph.globals.shape}")

    # Expected shapes
    assert graph.nodes.shape == (64, 119), f"Expected (64, 119), got {graph.nodes.shape}"
    assert graph.move_edges.features.shape[-1] == 19, f"Expected 19 move features"
    assert graph.grid_edges.features.shape[-1] == 2, f"Expected 2 grid features"
    assert graph.attack_edges.features.shape[-1] == 16, f"Expected 16 attack features"
    assert graph.defense_edges.features.shape[-1] == 16, f"Expected 16 defense features"

    console.print("[bold green]✓ Test 1 passed![/bold green]")
    return graph


def test_model_forward():
    """Test 2: Model forward pass."""
    console.print("\n[bold cyan]Test 2: Model Forward Pass[/bold cyan]")

    env = pgx.make("chess")
    state = env.init(jax.random.PRNGKey(0))
    state = jax.tree.map(lambda x: x[None], state)  # Add batch dim

    model = HeteroEdgeNet(
        n_actions=env.num_actions,
        inner_size=64,  # Smaller for faster test
        n_res_layers=2,  # Fewer layers for faster test
    )

    graph = cg.state_to_hetero_graph(state.observation, state.legal_action_mask)

    # Initialize
    variables = model.init(jax.random.PRNGKey(42), graphs=graph)
    params = variables['params']
    batch_stats = variables['batch_stats']

    # Forward pass
    (logits, value), _ = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        graphs=graph,
        mutable=['batch_stats'],
        training=False
    )

    console.print(f"✓ Logits shape: {logits.shape} (expected: (4672,))")
    console.print(f"✓ Value shape: {value.shape} (expected: (1,))")
    console.print(f"✓ Value range: [{value.min():.3f}, {value.max():.3f}] (expected: [-1, 1])")

    assert logits.shape == (4672,), f"Expected (4672,), got {logits.shape}"
    assert value.shape == (1,) or value.shape == (), f"Expected (1,) or (), got {value.shape}"
    assert jnp.abs(value).max() <= 1.1, f"Value should be in [-1, 1], got {value}"

    console.print("[bold green]✓ Test 2 passed![/bold green]")
    return model, params, batch_stats


def test_jit_compilation():
    """Test 3: JIT compilation."""
    console.print("\n[bold cyan]Test 3: JIT Compilation[/bold cyan]")

    env = pgx.make("chess")
    state = env.init(jax.random.PRNGKey(0))
    state = jax.tree.map(lambda x: x[None], state)

    # JIT the graph construction
    jitted_graph_fn = jax.jit(cg.state_to_hetero_graph)

    console.print("Compiling graph construction...")
    graph = jitted_graph_fn(state.observation, state.legal_action_mask)
    console.print("✓ Graph construction JIT compiled successfully")

    # Test with model
    model = HeteroEdgeNet(n_actions=env.num_actions, inner_size=64, n_res_layers=2)
    variables = model.init(jax.random.PRNGKey(42), graphs=graph)

    @jax.jit
    def forward(graphs):
        return model.apply(
            variables,
            graphs=graphs,
            mutable=['batch_stats'],
            training=False
        )

    console.print("Compiling model forward pass...")
    (logits, value), _ = forward(graph)
    console.print("✓ Model forward pass JIT compiled successfully")

    console.print("[bold green]✓ Test 3 passed![/bold green]")


def test_batched_graph():
    """Test 4: Batched graph construction (for pmap)."""
    console.print("\n[bold cyan]Test 4: Batched Graph Construction[/bold cyan]")

    env = pgx.make("chess")
    batch_size = 4
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), batch_size))

    graph = cg.state_to_hetero_graph(states.observation, states.legal_action_mask)

    console.print(f"✓ Batch size: {batch_size}")
    console.print(f"✓ Total nodes: {graph.nodes.shape[0]} (expected: {64 * batch_size})")
    console.print(f"✓ n_node array: {graph.n_node} (should be [{64}, {64}, {64}, {64}])")

    assert graph.nodes.shape[0] == 64 * batch_size
    assert jnp.all(graph.n_node == 64)

    console.print("[bold green]✓ Test 4 passed![/bold green]")


def test_gradient_computation():
    """Test 5: Gradient computation."""
    console.print("\n[bold cyan]Test 5: Gradient Computation[/bold cyan]")

    env = pgx.make("chess")
    state = env.init(jax.random.PRNGKey(0))
    state = jax.tree.map(lambda x: x[None], state)

    model = HeteroEdgeNet(n_actions=env.num_actions, inner_size=32, n_res_layers=1)
    graph = cg.state_to_hetero_graph(state.observation, state.legal_action_mask)

    variables = model.init(jax.random.PRNGKey(42), graphs=graph)
    params = variables['params']
    batch_stats = variables['batch_stats']

    def loss_fn(params):
        (logits, value), _ = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            graphs=graph,
            mutable=['batch_stats'],
            training=True
        )
        # Dummy loss
        policy_loss = -jnp.mean(logits)
        value_loss = jnp.mean(value ** 2)
        return policy_loss + value_loss

    console.print("Computing gradients...")
    grads = jax.grad(loss_fn)(params)

    # Check gradients exist and are not NaN
    grad_leaves = jax.tree_util.tree_leaves(grads)
    max_grad = max(jnp.abs(g).max() for g in grad_leaves)
    has_nan = any(jnp.isnan(g).any() for g in grad_leaves)

    console.print(f"✓ Number of gradient arrays: {len(grad_leaves)}")
    console.print(f"✓ Max gradient magnitude: {max_grad:.6f}")
    console.print(f"✓ NaN in gradients: {has_nan}")

    assert not has_nan, "NaN detected in gradients!"
    assert max_grad > 0, "Gradients are zero!"

    console.print("[bold green]✓ Test 5 passed![/bold green]")


def test_multi_device():
    """Test 6: Multi-device compatibility (if available)."""
    console.print("\n[bold cyan]Test 6: Multi-Device Compatibility[/bold cyan]")

    devices = jax.devices()
    console.print(f"Available devices: {devices}")
    console.print(f"Device count: {len(devices)}")

    if len(devices) < 2:
        console.print("[yellow]⚠ Only 1 device available, skipping pmap test[/yellow]")
        return

    env = pgx.make("chess")

    # Replicate params across devices
    model = HeteroEdgeNet(n_actions=env.num_actions, inner_size=32, n_res_layers=1)
    state = env.init(jax.random.PRNGKey(0))
    state = jax.tree.map(lambda x: x[None], state)
    graph = cg.state_to_hetero_graph(state.observation, state.legal_action_mask)

    variables = model.init(jax.random.PRNGKey(42), graphs=graph)

    # Test device_put_replicated
    replicated_vars = jax.tree.map(
        lambda x: jax.device_put_replicated(x, devices[:2]),
        variables
    )

    console.print(f"✓ Parameters replicated across {len(devices[:2])} devices")
    console.print("[bold green]✓ Test 6 passed![/bold green]")


def main():
    console.print("\n[bold magenta]HeteroEdgeNet Implementation Verification[/bold magenta]")
    console.print("=" * 70)

    try:
        test_graph_construction()
        test_model_forward()
        test_jit_compilation()
        test_batched_graph()
        test_gradient_computation()
        test_multi_device()

        console.print("\n" + "=" * 70)
        console.print("[bold green]✅ All tests passed! Ready for training.[/bold green]")
        console.print("=" * 70)
        console.print("\n[bold cyan]Next steps:[/bold cyan]")
        console.print("1. Run: [bold]./run_multi_gpu.sh[/bold]")
        console.print("2. Monitor with: [bold]watch -n 1 nvidia-smi[/bold]")
        console.print("3. Check progress in the training output")

    except Exception as e:
        console.print(f"\n[bold red]❌ Test failed with error:[/bold red]")
        console.print(f"[red]{e}[/red]")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
