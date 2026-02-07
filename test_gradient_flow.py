#!/usr/bin/env python3
"""Test gradient flow through MemoTokenMemoryEncoder without full training pipeline."""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add skyrl-train to path
sys.path.insert(0, str(Path(__file__).parent / "skyrl-train"))

from skyrl_train.examples.modalities.memo_handlers import MemoTokenMemoryEncoder


def test_gradient_flow():
    """Test that gradients flow through memory module but not embedding."""
    print("=" * 60)
    print("Testing MemoTokenMemoryEncoder gradient flow")
    print("=" * 60)

    # Create encoder
    print("\n1. Creating MemoTokenMemoryEncoder...")
    encoder = MemoTokenMemoryEncoder(
        modality_id="test_memory",
        role="encoder",
        model_path="Qwen/Qwen3-4B-Instruct-2507",
        embedding_dim=2560,
        num_memories=8,
        output_dim=2560,
        num_heads=8,
        num_layers=1,
        dropout=0.1,
        memory_init="xavier_uniform",
        max_doc_tokens=256,
        memo_repo_root="/home/parsaidp/MeMo",
    )

    # Check parameters
    print("\n2. Checking parameters() and named_parameters()...")
    encoder_params = list(encoder.parameters())
    memory_params = list(encoder.memory.parameters())
    encoder_named_params = list(encoder.named_parameters())
    memory_named_params = list(encoder.memory.named_parameters())

    print(f"   - encoder.parameters() count: {len(encoder_params)}")
    print(f"   - encoder.memory.parameters() count: {len(memory_params)}")
    print(f"   - Embedding excluded from parameters(): {len(encoder_params) == len(memory_params)}")
    print(f"   - encoder.named_parameters() count: {len(encoder_named_params)}")
    print(f"   - encoder.memory.named_parameters() count: {len(memory_named_params)}")
    print(f"   - Embedding excluded from named_parameters(): {len(encoder_named_params) == len(memory_named_params)}")

    # Check requires_grad
    print("\n3. Checking requires_grad flags...")
    print(f"   - Embedding weight requires_grad: {encoder.embedding.weight.requires_grad}")
    print(f"   - Memory params require_grad: {all(p.requires_grad for p in memory_params)}")

    # Simulate forward pass
    print("\n4. Running forward pass...")
    dummy_token_ids = torch.randint(0, 50000, (128,))  # 128 tokens
    payloads = [{"token_ids": dummy_token_ids.tolist()}]

    memory_embeddings_list = encoder.encode(payloads)
    memory_embeddings = memory_embeddings_list[0]  # Shape: [num_memories, output_dim]

    print(f"   - Input tokens: {dummy_token_ids.shape}")
    print(f"   - Memory embeddings: {memory_embeddings.shape}")
    print(f"   - Has grad_fn: {memory_embeddings.grad_fn is not None}")

    # Compute dummy loss and backward
    print("\n5. Computing loss and backward pass...")
    loss = memory_embeddings.mean()
    print(f"   - Loss: {loss.item():.6f}")

    try:
        loss.backward()
        print("   ✓ Backward pass succeeded!")
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        return False

    # Check gradients
    print("\n6. Checking gradients after backward...")
    embedding_has_grad = encoder.embedding.weight.grad is not None
    memory_has_grad = all(p.grad is not None for p in memory_params)

    print(f"   - Embedding weight has grad: {embedding_has_grad}")
    print(f"   - Memory params have grad: {memory_has_grad}")

    # Create optimizer and check what it will update
    print("\n7. Testing optimizer behavior...")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

    param_groups = optimizer.param_groups[0]['params']
    param_ids = {id(p) for p in param_groups}
    print(f"   - Optimizer has {len(param_groups)} parameters")
    print(f"   - Embedding in optimizer: {id(encoder.embedding.weight) in param_ids}")
    print(f"   - Memory params in optimizer: {all(id(p) in param_ids for p in memory_params)}")

    # Test optimizer step
    print("\n8. Testing optimizer step...")
    embedding_before = encoder.embedding.weight.data.clone()
    memory_param_before = next(encoder.memory.parameters()).data.clone()

    optimizer.step()

    embedding_changed = not torch.equal(embedding_before, encoder.embedding.weight.data)
    memory_changed = not torch.equal(memory_param_before, next(encoder.memory.parameters()).data)

    print(f"   - Embedding changed after step: {embedding_changed}")
    print(f"   - Memory params changed after step: {memory_changed}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    success = True
    checks = {
        "Parameters override works": len(encoder_params) == len(memory_params),
        "Named parameters override works": len(encoder_named_params) == len(memory_named_params),
        "Embedding has requires_grad": encoder.embedding.weight.requires_grad,
        "Memory has requires_grad": all(p.requires_grad for p in memory_params),
        "Forward produces grad_fn": memory_embeddings.grad_fn is not None,
        "Backward pass succeeds": True,  # We got here
        "Memory gets gradients": memory_has_grad,
        "Embedding NOT in optimizer": id(encoder.embedding.weight) not in param_ids,
        "Memory params in optimizer": all(id(p) in param_ids for p in memory_params),
        "Memory params updated": memory_changed,
        "Embedding NOT updated": not embedding_changed,
    }

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
        success = success and passed

    print("=" * 60)
    if success:
        print("🎉 ALL CHECKS PASSED - Gradient flow is working correctly!")
    else:
        print("❌ SOME CHECKS FAILED - Gradient flow has issues")
    print("=" * 60)

    return success


if __name__ == "__main__":
    try:
        success = test_gradient_flow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
