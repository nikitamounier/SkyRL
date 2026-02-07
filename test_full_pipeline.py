#!/usr/bin/env python3
"""Test the FULL pipeline: parameters(), _freeze_base_parameters(), and gradient flow."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "skyrl-train"))

from skyrl_train.examples.modalities.memo_handlers import MemoTokenMemoryEncoder


def test_full_pipeline():
    """Simulate the EXACT flow that happens in model_wrapper and training."""
    print("=" * 60)
    print("Testing FULL training pipeline simulation")
    print("=" * 60)

    # Step 1: Create encoder (simulates ModalitiesManager.__init__)
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

    print(f"   Initial state:")
    print(f"   - embedding.requires_grad: {encoder.embedding.weight.requires_grad}")
    print(f"   - memory params requires_grad: {all(p.requires_grad for p in encoder.memory.parameters())}")
    print(f"   - encoder.parameters() count: {len(list(encoder.parameters()))}")
    print(f"   - encoder.named_parameters() count: {len(list(encoder.named_parameters()))}")

    # Step 2: _configure_module_trainability (simulates ModalitiesManager._configure_module_trainability)
    print("\n2. Simulating _configure_module_trainability(trainable=True)...")
    trainable = True
    for param in encoder.parameters():  # This should only iterate over memory params!
        param.requires_grad = trainable

    print(f"   After _configure_module_trainability:")
    print(f"   - embedding.requires_grad: {encoder.embedding.weight.requires_grad}")
    print(f"   - memory params requires_grad: {all(p.requires_grad for p in encoder.memory.parameters())}")

    # Step 3: Simulate _freeze_base_parameters (model_wrapper._freeze_base_parameters)
    print("\n3. Simulating _freeze_base_parameters()...")
    # In the real code, this iterates over encoder.parameters() and sets requires_grad
    for param in encoder.parameters():
        param.requires_grad = trainable

    print(f"   After _freeze_base_parameters:")
    print(f"   - embedding.requires_grad: {encoder.embedding.weight.requires_grad}")
    print(f"   - memory params requires_grad: {all(p.requires_grad for p in encoder.memory.parameters())}")

    # Step 4: Check optimizer (simulates optimizer creation)
    print("\n4. Creating optimizer with encoder.parameters()...")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    param_count = len(optimizer.param_groups[0]['params'])
    embedding_in_optimizer = id(encoder.embedding.weight) in {id(p) for p in optimizer.param_groups[0]['params']}

    print(f"   - Optimizer param count: {param_count}")
    print(f"   - Embedding in optimizer: {embedding_in_optimizer}")

    # Step 5: Test forward pass and backward (simulates training step)
    print("\n5. Testing forward pass and backward...")
    dummy_token_ids = torch.randint(0, 50000, (128,))
    payloads = [{"token_ids": dummy_token_ids.tolist()}]

    memory_embeddings_list = encoder.encode(payloads)
    memory_embeddings = memory_embeddings_list[0]

    print(f"   - memory_embeddings.shape: {memory_embeddings.shape}")
    print(f"   - memory_embeddings.grad_fn: {memory_embeddings.grad_fn is not None}")

    loss = memory_embeddings.mean()

    try:
        loss.backward()
        print("   ✓ Backward pass SUCCEEDED!")
        backward_ok = True
    except Exception as e:
        print(f"   ✗ Backward pass FAILED: {e}")
        backward_ok = False

    # Step 6: Check gradients
    print("\n6. Checking gradients...")
    if backward_ok:
        embedding_has_grad = encoder.embedding.weight.grad is not None
        memory_has_grad = all(p.grad is not None for p in encoder.memory.parameters())
        print(f"   - embedding has grad: {embedding_has_grad}")
        print(f"   - memory has grad: {memory_has_grad}")

    # Summary
    print("\n" + "=" * 60)
    print("CRITICAL CHECKS:")
    print("=" * 60)

    checks = {
        "Embedding stays frozen (requires_grad=False)": not encoder.embedding.weight.requires_grad,
        "Memory params trainable (requires_grad=True)": all(p.requires_grad for p in encoder.memory.parameters()),
        "Embedding NOT in optimizer": not embedding_in_optimizer,
        "Backward pass succeeds": backward_ok,
    }

    success = all(checks.values())
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")

    print("=" * 60)
    if success:
        print("🎉 FULL PIPELINE TEST PASSED!")
        print("The fix should work with FSDP training!")
    else:
        print("❌ FULL PIPELINE TEST FAILED!")
        print("There are still issues that need to be fixed!")
    print("=" * 60)

    return success


if __name__ == "__main__":
    try:
        success = test_full_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
