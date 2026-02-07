#!/usr/bin/env python3
"""Test that embedding as buffer fixes all issues."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "skyrl-train"))

from skyrl_train.examples.modalities.memo_handlers import MemoTokenMemoryEncoder


def test_buffer_fix():
    print("=" * 60)
    print("Testing embedding as buffer fix")
    print("=" * 60)

    # Create encoder
    print("\n1. Creating MemoTokenMemoryEncoder...")
    encoder = MemoTokenMemoryEncoder(
        modality_id="memo_memory",
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

    # Check structure
    print("\n2. Checking module structure...")
    print(f"   - Has embedding_weight buffer: {hasattr(encoder, 'embedding_weight')}")
    print(f"   - Has embedding module: {hasattr(encoder, 'embedding')}")
    print(f"   - embedding_weight is buffer: {'embedding_weight' in dict(encoder.named_buffers())}")

    # Check parameters
    print("\n3. Checking parameters...")
    all_params = list(encoder.parameters())
    memory_params = list(encoder.memory.parameters())
    print(f"   - Total parameters: {len(all_params)}")
    print(f"   - Memory parameters: {len(memory_params)}")
    print(f"   - Parameters match (embedding excluded): {len(all_params) == len(memory_params)}")

    # Simulate _configure_module_trainability
    print("\n4. Simulating _configure_module_trainability...")
    for param in encoder.parameters():
        param.requires_grad = True
    print(f"   - Memory params have requires_grad=True: {all(p.requires_grad for p in encoder.memory.parameters())}")

    # Check optimizer
    print("\n5. Creating optimizer...")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    param_count = len(optimizer.param_groups[0]['params'])
    print(f"   - Optimizer has {param_count} parameters")
    print(f"   - Matches memory param count: {param_count == len(memory_params)}")

    # Test forward pass
    print("\n6. Testing forward pass...")
    dummy_token_ids = torch.randint(0, 50000, (128,))
    payloads = [{"token_ids": dummy_token_ids.tolist()}]

    memory_embeddings_list = encoder.encode(payloads)
    memory_embeddings = memory_embeddings_list[0]

    print(f"   - Output shape: {memory_embeddings.shape}")
    print(f"   - Output has grad_fn: {memory_embeddings.grad_fn is not None}")

    # Test backward
    print("\n7. Testing backward pass...")
    loss = memory_embeddings.mean()

    try:
        loss.backward()
        print("   ✓ Backward SUCCEEDED!")
        backward_ok = True
    except Exception as e:
        print(f"   ✗ Backward FAILED: {e}")
        backward_ok = False
        return False

    # Check gradients
    print("\n8. Checking gradients...")
    memory_has_grad = all(p.grad is not None for p in encoder.memory.parameters())
    print(f"   - Memory params have gradients: {memory_has_grad}")

    # Test optimizer step
    print("\n9. Testing optimizer step...")
    memory_param_before = next(encoder.memory.parameters()).data.clone()
    embedding_before = encoder.embedding_weight.data.clone()

    optimizer.step()

    memory_changed = not torch.equal(memory_param_before, next(encoder.memory.parameters()).data)
    embedding_changed = not torch.equal(embedding_before, encoder.embedding_weight.data)

    print(f"   - Memory params updated: {memory_changed}")
    print(f"   - Embedding buffer unchanged: {not embedding_changed}")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL CHECKS:")
    print("=" * 60)

    checks = {
        "Embedding is buffer (not parameter)": 'embedding_weight' in dict(encoder.named_buffers()),
        "Only memory params in parameters()": len(all_params) == len(memory_params),
        "Only memory params in optimizer": param_count == len(memory_params),
        "Forward pass works": True,
        "Output has grad_fn": memory_embeddings.grad_fn is not None,
        "Backward pass succeeds": backward_ok,
        "Memory params get gradients": memory_has_grad,
        "Memory params updated": memory_changed,
        "Embedding buffer unchanged": not embedding_changed,
    }

    success = all(checks.values())
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")

    print("=" * 60)
    if success:
        print("🎉 BUFFER FIX WORKS PERFECTLY!")
        print("Ready for FSDP training!")
    else:
        print("❌ Buffer fix has issues")
    print("=" * 60)

    return success


if __name__ == "__main__":
    try:
        success = test_buffer_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
