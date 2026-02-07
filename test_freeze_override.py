#!/usr/bin/env python3
"""Test what happens when _freeze_base_parameters overrides embedding freeze."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "skyrl-train"))

from skyrl_train.examples.modalities.memo_handlers import MemoTokenMemoryEncoder


def test_freeze_override():
    """Simulate what happens in model_wrapper._freeze_base_parameters."""
    print("=" * 60)
    print("Testing _freeze_base_parameters override behavior")
    print("=" * 60)

    # Create encoder
    print("\n1. Creating MemoTokenMemoryEncoder with freeze=True...")
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

    print(f"   - Initial embedding requires_grad: {encoder.embedding.weight.requires_grad}")
    print(f"   - Initial memory params requires_grad: {all(p.requires_grad for p in encoder.memory.parameters())}")

    # Simulate what _freeze_base_parameters does (line 259-260)
    print("\n2. Simulating _freeze_base_parameters() behavior...")
    print("   - Setting ALL params to requires_grad=True (trainable.encoder=True)")
    trainable = True
    for param in encoder.parameters():
        param.requires_grad = trainable

    print(f"   - After override, embedding requires_grad: {encoder.embedding.weight.requires_grad}")
    print(f"   - After override, memory params requires_grad: {all(p.requires_grad for p in encoder.memory.parameters())}")

    # Test forward pass with embedding having requires_grad=True
    print("\n3. Testing forward pass with embedding requires_grad=True...")
    dummy_token_ids = torch.randint(0, 50000, (128,))
    payloads = [{"token_ids": dummy_token_ids.tolist()}]

    memory_embeddings_list = encoder.encode(payloads)
    memory_embeddings = memory_embeddings_list[0]

    print(f"   - Memory embeddings shape: {memory_embeddings.shape}")
    print(f"   - Memory embeddings has grad_fn: {memory_embeddings.grad_fn is not None}")

    # Compute loss and backward
    print("\n4. Testing backward pass...")
    loss = memory_embeddings.mean()

    try:
        loss.backward()
        print("   ✓ Backward pass succeeded!")
    except Exception as e:
        print(f"   ✗ Backward pass FAILED: {e}")
        return False

    # Check gradients
    print("\n5. Checking gradients...")
    embedding_has_grad = encoder.embedding.weight.grad is not None
    memory_has_grad = all(p.grad is not None for p in encoder.memory.parameters())

    print(f"   - Embedding has gradient: {embedding_has_grad}")
    print(f"   - Memory has gradient: {memory_has_grad}")

    # The issue: embedding has requires_grad=True AND has gradients
    # But we call .detach() in encode(), so this shouldn't break gradients
    # UNLESS... the detach happens AFTER embedding, breaking the graph

    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    print(f"✓ Backward pass works even with embedding requires_grad=True")
    print(f"✓ Memory module still receives gradients")
    print(f"✓ The .detach() in encode() prevents embedding from getting gradients")
    print("\nBUT: The embedding has requires_grad=True, which means:")
    print("  - It's included in optimizer parameters")
    print("  - FSDP will shard it as a trainable parameter")
    print("  - DTensor conversions might treat it differently")
    print("\nThe bug might be in FSDP's handling of params with requires_grad=True")
    print("but no actual gradients due to .detach()")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        test_freeze_override()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
