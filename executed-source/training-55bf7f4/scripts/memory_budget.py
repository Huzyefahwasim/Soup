"""Calculate a pre-run envelope; these numbers are estimates, not measurements."""
import argparse
import json
from pathlib import Path


def estimate(c, batch=1, sequence=512, rank=8):
    h, i, n, v = (c[k] for k in ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'vocab_size'))
    kv = h * c['num_key_value_heads'] // c['num_attention_heads']
    layer = 2*h*h + 2*h*kv + 3*h*i + 2*h + h + 2*kv
    lora = n * rank * (3*h + kv)  # A+B for q_proj and v_proj
    embedding = v*h*2*(1 if c.get('tie_word_embeddings') else 2)
    elements = 2*batch*sequence*v
    parts = {
        'resident_embedding_and_head': embedding,
        'two_streaming_layer_buffers': 2*layer*2,
        'fp32_adapter_grad_and_adam_states': lora*16,
        'frozen_reference_adapter': lora*4,
        'checkpoint_boundaries': 2*batch*sequence*h*2*(n+1),
        'logits_and_loss_workspace_envelope': elements*14,
        'within_layer_activations_allowance': 2*batch*sequence*i*2*8,
        'cuda_context_and_allocator_allowance': 1024**3,
    }
    return {'assumptions': {'batch_pairs': batch, 'concatenated_rows': 2*batch,
            'max_sequence': sequence, 'base_bytes_per_parameter': 2, 'rank': rank,
            'quantization': 'none', 'reference': 'sequential shared base, frozen initial LoRA'},
            'layer_parameters': layer, 'lora_parameters': lora,
            'logit_elements': elements, 'parts_bytes': parts,
            'conservative_total_gib': sum(parts.values())/1024**3,
            'host_decoder_store_gib': n*layer*2/1024**3,
            'full_fp16_weights_gib': (n*layer*2+embedding)/1024**3,
            'caveats': ['14 bytes per logit is an envelope for overlapping logits, float32 loss operations and temporary tensors, not a measured allocation.',
                        'The reference pass adds compute and transient activations; its base weights are shared. It does not add a second full model.',
                        'Streaming retains host weights and checkpoints. Free Colab RAM and disk must also fit.',
                        'nvidia-smi reports driver allocations; allocated/reserved PyTorch peaks and sampled driver peaks measure different things.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Pinned model config.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = estimate(json.loads(Path(args.config).read_text()))
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
