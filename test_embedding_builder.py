#!/usr/bin/env python3
"""Test the full embedding builder pipeline with modality replacements."""

import torch
import sys
from pathlib import Path

# Add skyrl-train to path
sys.path.insert(0, str(Path(__file__).parent / "skyrl-train"))

from skyrl_train.examples.modalities.memo_handlers import MemoTokenMemoryEncoder
from skyrl_train.modalities.embedding_builder import PromptEmbeddingBuilder, EmbeddingSpan


def test_embedding_builder_gradient_flow():
    """Test gradient flow through the full embedding builder pipeline."""
    print("=" * 60)
    print("Testing PromptEmbeddingBuilder + MemoTokenMemoryEncoder")
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

    # Create embedding builder
    print("\n2. Creating PromptEmbeddingBuilder...")
    embedding_builder = PromptEmbeddingBuilder(
        embedding_dim=2560,
        target_device=torch.device("cpu"),
        target_dtype=torch.bfloat16,
    )
    embedding_builder.ensure_initialized("Qwen/Qwen3-4B-Instruct-2507")
    print(f"   - Base embedding table loaded: {embedding_builder.has_base_embedding()}")

    # Create dummy data
    print("\n3. Preparing test data...")
    # Prompt tokens with placeholder for modality
    prompt_token_ids = [[1, 2, 3, 999, 999, 999, 999, 4, 5, 6]]  # 999s are placeholders

    # Generate modality embeddings
    dummy_token_ids = torch.randint(0, 50000, (128,))
    payloads = [{"token_ids": dummy_token_ids.tolist()}]
    modality_embeddings_list = encoder.encode(payloads)
    modality_embeddings = modality_embeddings_list[0]  # [8, 2560]

    print(f"   - Prompt tokens: {prompt_token_ids}")
    print(f"   - Modality embeddings shape: {modality_embeddings.shape}")
    print(f"   - Modality embeddings has grad_fn: {modality_embeddings.grad_fn is not None}")

    # Create modality replacements (replace tokens 3-6 with modality embeddings)
    modality_replacements = {
        0: [  # sample_idx 0
            (EmbeddingSpan(start=3, length=4), modality_embeddings[:4])  # Use first 4 memory slots
        ]
    }

    # Build prompt embeddings
    print("\n4. Building prompt embeddings with modality replacements...")
    prompt_embeddings_list = embedding_builder.build_prompt_embeddings_for_batch(
        prompt_token_ids,
        modality_replacements
    )
    prompt_embeddings = prompt_embeddings_list[0]  # [seq_len, hidden_dim]

    print(f"   - Prompt embeddings shape: {prompt_embeddings.shape}")
    print(f"   - Prompt embeddings dtype: {prompt_embeddings.dtype}")
    print(f"   - Prompt embeddings has grad_fn: {prompt_embeddings.grad_fn is not None}")

    # Check gradient flow
    print("\n5. Testing gradient flow through full pipeline...")
    if prompt_embeddings.grad_fn is None:
        print("   ✗ ERROR: Final embeddings have no grad_fn!")
        return False

    # Compute loss and backward
    loss = prompt_embeddings.mean()
    print(f"   - Loss: {loss.item():.6f}")

    try:
        loss.backward()
        print("   ✓ Backward pass succeeded!")
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        return False

    # Check gradients
    print("\n6. Checking gradients...")
    memory_params = list(encoder.memory.parameters())
    memory_has_grad = all(p.grad is not None for p in memory_params)
    embedding_has_grad = encoder.embedding.weight.grad is not None

    print(f"   - Memory params have gradients: {memory_has_grad}")
    print(f"   - Embedding has gradients: {embedding_has_grad}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    checks = {
        "Modality embeddings have grad_fn": modality_embeddings.grad_fn is not None,
        "Final embeddings have grad_fn": prompt_embeddings.grad_fn is not None,
        "Backward pass succeeds": True,
        "Memory gets gradients": memory_has_grad,
    }

    success = True
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
        success = success and passed

    print("=" * 60)
    if success:
        print("🎉 FULL PIPELINE TEST PASSED!")
    else:
        print("❌ FULL PIPELINE TEST FAILED!")
    print("=" * 60)

    return success


if __name__ == "__main__":
    try:
        success = test_embedding_builder_gradient_flow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
