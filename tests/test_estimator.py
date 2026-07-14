from sbench.adapters import resolve_adapter
from sbench.components import AttentionComponent, CacheComponent, MoEComponent
from sbench.descriptor import ArchitectureDescriptor, AttentionDescriptor, CacheDescriptor, FFNDescriptor, MoEDescriptor, RuntimeDescriptor, descriptor_from_config
from sbench.estimator import estimate_records


def test_qwen3_descriptor_counts_moe_layers():
    desc = descriptor_from_config({
        "num_hidden_layers": 4,
        "hidden_size": 16,
        "intermediate_size": 64,
        "moe_intermediate_size": 8,
        "mlp_only_layers": [0],
        "decoder_sparse_step": 2,
        "num_experts": 128,
        "num_experts_per_tok": 8,
    }, model_name="Qwen/Qwen3-30B-A3B")
    assert desc.moe.moe_layers == 2
    assert desc.ffn.dense_layers == 2
    assert desc.moe.top_k == 8


def test_deepseek_descriptor_detects_mla_cache():
    desc = descriptor_from_config({
        "num_hidden_layers": 6,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "kv_lora_rank": 12,
        "moe_intermediate_size": 8,
        "intermediate_size": 64,
        "first_k_dense_replace": 2,
    }, model_name="deepseek-ai/DeepSeek-V3")
    assert desc.attention.type == "mla"
    assert desc.cache.type == "latent_kv"
    assert desc.ffn.dense_layers == 2
    assert desc.moe.moe_layers == 4


def test_overrides_force_linear_attention_and_state_cache():
    desc = descriptor_from_config(
        {"num_hidden_layers": 2, "hidden_size": 16, "num_attention_heads": 2},
        overrides={"architecture": {"attention": {"type": "linear"}, "cache": {"type": "recurrent_state", "recurrent_state_size": 128}}},
    )
    assert desc.attention.type == "linear"
    assert desc.cache.type == "recurrent_state"
    assert CacheComponent().per_token_units(desc) == 256


def test_cache_components_match_reference_shapes():
    kv = ArchitectureDescriptor(cache=CacheDescriptor(type="kv", num_layers=3, head_dim=8, num_key_value_heads=2))
    mla = ArchitectureDescriptor(cache=CacheDescriptor(type="latent_kv", num_layers=3, kv_lora_rank=12, qk_rope_head_dim=8))
    assert CacheComponent().per_token_units(kv) == 2 * 3 * 8 * 2
    assert CacheComponent().per_token_units(mla) == 3 * (12 + 8)


def test_sparse_attention_scales_attention_score():
    dense = ArchitectureDescriptor(attention=AttentionDescriptor(type="gqa", num_layers=2, hidden_size=16, num_attention_heads=2, num_key_value_heads=1, head_dim=4))
    sparse = ArchitectureDescriptor(attention=AttentionDescriptor(type="sparse", num_layers=2, hidden_size=16, num_attention_heads=2, num_key_value_heads=1, head_dim=4, sparse_factor=0.25))
    record = {"forward_mode": "prefill", "seq_lens_sum": 100, "batch_size": 1}
    assert AttentionComponent().attention_score_units(sparse, record) == AttentionComponent().attention_score_units(dense, record) * 0.25


def test_prefill_attention_score_scales_with_context_mass():
    dense = ArchitectureDescriptor(attention=AttentionDescriptor(type="gqa", num_layers=1, hidden_size=16, num_attention_heads=2, num_key_value_heads=1, head_dim=4))
    short = {"forward_mode": "prefill", "seq_lens_sum": 100, "batch_size": 1, "processed_tokens": 10}
    long = {"forward_mode": "prefill", "seq_lens_sum": 200, "batch_size": 1, "processed_tokens": 10}
    assert AttentionComponent().attention_score_units(dense, long) > AttentionComponent().attention_score_units(dense, short)


def test_moe_bandwidth_uses_activation_but_flops_do_not():
    desc = ArchitectureDescriptor(moe=MoEDescriptor(enabled=True, moe_layers=2, hidden_size=10, expert_intermediate_size=5, shared_experts=1))
    cost = MoEComponent().estimate(desc, {"expert_activation": 3})
    expert = 5 * 3 * 10 / 1e12
    assert cost.bandwidth_units == 2 * (3 * expert + expert)
    assert cost.flops_units == 2 * (expert + expert)


def test_estimator_preserves_packed_prefill_throughput_and_processed_tokens():
    desc = ArchitectureDescriptor(
        attention=AttentionDescriptor(type="gqa", num_layers=1, hidden_size=1, num_attention_heads=1, num_key_value_heads=1, head_dim=1),
        cache=CacheDescriptor(type="kv", num_layers=1, head_dim=1, num_key_value_heads=1),
        runtime=RuntimeDescriptor(precision_bytes=2, num_gpus=1, peak_bandwidth_tb=1, peak_flops_tf=1),
    )
    result = estimate_records(desc, [{"forward_mode": "prefill", "latency": 2.0, "seq_lens_sum": 400, "batch_size": 4, "processed_tokens": 100, "expert_activation": 0}])
    assert result.prefill_tp == 200
    assert result.kv_size == 2e-4


def test_adapter_registry_selects_component_mix():
    result = resolve_adapter({"kv_lora_rank": 12, "qk_rope_head_dim": 8, "num_hidden_layers": 2}, model_name="deepseek-ai/DeepSeek-V3")
    assert result.name == "deepseek_mla"
    result = resolve_adapter({"moe_intermediate_size": 8}, model_name="Qwen/Qwen3-Next-80B-A3B", overrides={"architecture": {"attention": {"type": "hybrid"}}})
    assert result.name == "qwen_hybrid"


def test_estimator_keeps_timing_only_negative_activation_records():
    desc = ArchitectureDescriptor(
        attention=AttentionDescriptor(type="gqa", num_layers=1, hidden_size=8, num_attention_heads=1, num_key_value_heads=1, head_dim=8),
        cache=CacheDescriptor(type="kv", num_layers=1, head_dim=8, num_key_value_heads=1),
        runtime=RuntimeDescriptor(precision_bytes=2, num_gpus=1, peak_bandwidth_tb=1, peak_flops_tf=1),
    )
    result = estimate_records(desc, [{"forward_mode": "decode", "latency": 1.0, "seq_lens_sum": 100, "batch_size": 4, "expert_activation": -1}])
    assert result.decoding_throughput == 4
    assert result.decoding_smfu > 0
