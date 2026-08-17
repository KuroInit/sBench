from sbench.adapters import resolve_adapter
from sbench.components import AttentionComponent, CacheComponent, MoEComponent, RouterComponent, prefill_context_mass
from sbench.descriptor import ArchitectureDescriptor, AttentionDescriptor, CacheDescriptor, FFNDescriptor, MoEDescriptor, RuntimeDescriptor, descriptor_from_config
from sbench.estimator import estimate_records
import pytest
from sbench.moe_cap_estimator import estimate_moe_cap_compatible, support_status


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


@pytest.mark.parametrize(
    ("model_name", "config", "adapter_name", "attention_type", "moe_enabled", "moe_cap_supported"),
    [
        ("Qwen/Qwen3-8B", {"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096, "intermediate_size": 12288, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}, "dense_transformer", "gqa", False, False),
        ("Qwen/Qwen3-14B", {"model_type": "qwen3", "num_hidden_layers": 40, "hidden_size": 5120, "intermediate_size": 17408, "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128}, "dense_transformer", "gqa", False, False),
        ("Qwen/Qwen3-32B", {"model_type": "qwen3", "num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 25600, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128}, "dense_transformer", "gqa", False, False),
        ("Qwen/Qwen3-30B-A3B", {"model_type": "qwen3_moe", "num_hidden_layers": 48, "hidden_size": 2048, "intermediate_size": 6144, "moe_intermediate_size": 768, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8, "decoder_sparse_step": 1, "mlp_only_layers": []}, "qwen_moe", "gqa", True, True),
        ("Qwen/Qwen3.5-9B", {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 12288, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 32, "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8}}, "qwen_hybrid", "hybrid", False, False),
        ("Qwen/Qwen3.6-27B", {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 17408, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 48, "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16}}, "qwen_hybrid", "hybrid", False, False),
        ("moonshotai/Moonlight-16B-A3B", {"model_type": "deepseek_v3", "num_hidden_layers": 27, "hidden_size": 2048, "intermediate_size": 11264, "moe_intermediate_size": 1408, "num_attention_heads": 16, "num_key_value_heads": 16, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128, "first_k_dense_replace": 1, "n_routed_experts": 64, "n_shared_experts": 2, "num_experts_per_tok": 6}, "deepseek_mla", "mla", True, False),
    ],
)
def test_supported_qwen_model_configs_resolve_to_expected_estimator_path(model_name, config, adapter_name, attention_type, moe_enabled, moe_cap_supported):
    result = resolve_adapter(config, model_name=model_name)
    assert result.name == adapter_name
    assert result.descriptor.attention.type == attention_type
    assert result.descriptor.moe.enabled is moe_enabled
    assert support_status(result.descriptor).supported is moe_cap_supported


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
    desc = ArchitectureDescriptor(moe=MoEDescriptor(enabled=True, moe_layers=2, hidden_size=10, expert_intermediate_size=5, shared_experts=1, top_k=2))
    cost = MoEComponent().estimate(desc, {"expert_activation": 3})
    expert = 5 * 3 * 10 / 1e12
    assert cost.bandwidth_units == 2 * (3 * expert + expert)
    assert cost.flops_units == 2 * (2 * expert + expert)



def test_prefill_context_mass_uses_probe_total_len():
    record = {
        "forward_mode": "prefill",
        "seq_lens_sum": 300,
        "processed_tokens": 30,
        "per_req_info": [
            {"extend_len": 10, "total_len": 100},
            {"extend_len": 20, "total_len": 200},
        ],
    }
    assert prefill_context_mass(record) == 10 * 100 + 20 * 200


def test_router_component_returns_per_token_units():
    desc = ArchitectureDescriptor(moe=MoEDescriptor(enabled=True, moe_layers=2, hidden_size=10, routed_experts=4))
    cost = RouterComponent().estimate(desc, {"forward_mode": "prefill", "processed_tokens": 100, "batch_size": 10})
    assert cost.flops_units == 2 * 10 * 4 / 1e12

def test_estimator_preserves_packed_prefill_throughput_and_processed_tokens():
    desc = ArchitectureDescriptor(
        attention=AttentionDescriptor(type="gqa", num_layers=1, hidden_size=1, num_attention_heads=1, num_key_value_heads=1, head_dim=1),
        cache=CacheDescriptor(type="kv", num_layers=1, head_dim=1, num_key_value_heads=1),
        runtime=RuntimeDescriptor(precision_bytes=2, num_gpus=1, peak_bandwidth_tb=1, peak_flops_tf=1),
    )
    result = estimate_records(desc, [{"forward_mode": "prefill", "latency": 2.0, "seq_lens_sum": 400, "batch_size": 4, "processed_tokens": 100, "per_req_info": [{"extend_len": 25, "total_len": 100}] * 4}])
    assert result.prefill_tp == 200
    assert result.kv_size == 2e-4


def test_estimator_rejects_multi_request_prefill_without_per_request_context():
    desc = ArchitectureDescriptor(
        attention=AttentionDescriptor(type="gqa", num_layers=1, hidden_size=1, num_attention_heads=1, num_key_value_heads=1, head_dim=1),
        cache=CacheDescriptor(type="kv", num_layers=1, head_dim=1, num_key_value_heads=1),
    )
    result = estimate_records(desc, [{"forward_mode": "prefill", "latency": 1.0, "seq_lens_sum": 100, "batch_size": 2, "processed_tokens": 100}])
    assert result.prefill_tp == 0


def test_model_name_does_not_imply_hybrid_attention_without_config_evidence():
    desc = descriptor_from_config(
        {"num_hidden_layers": 2, "hidden_size": 16, "num_attention_heads": 2},
        model_name="Qwen/Qwen3-Next-Example",
    )
    assert desc.attention.type == "mha"


def test_adapter_registry_selects_component_mix():
    result = resolve_adapter({"kv_lora_rank": 12, "qk_rope_head_dim": 8, "num_hidden_layers": 2}, model_name="deepseek-ai/DeepSeek-V3")
    assert result.name == "deepseek_mla"
    result = resolve_adapter({"moe_intermediate_size": 8}, model_name="Qwen/Qwen3-Next-80B-A3B", overrides={"architecture": {"attention": {"type": "hybrid"}}})
    assert result.name == "qwen_hybrid"


def test_estimator_keeps_timing_only_records_for_dense_models():
    desc = ArchitectureDescriptor(
        attention=AttentionDescriptor(type="gqa", num_layers=1, hidden_size=8, num_attention_heads=1, num_key_value_heads=1, head_dim=8),
        cache=CacheDescriptor(type="kv", num_layers=1, head_dim=8, num_key_value_heads=1),
        runtime=RuntimeDescriptor(precision_bytes=2, num_gpus=1, peak_bandwidth_tb=1, peak_flops_tf=1),
    )
    result = estimate_records(desc, [{"forward_mode": "decode", "latency": 1.0, "seq_lens_sum": 100, "batch_size": 4}])
    assert result.decoding_throughput == 4
    assert result.decoding_smfu > 0


def test_component_wise_moe_requires_real_activation():
    desc = ArchitectureDescriptor(moe=MoEDescriptor(enabled=True, moe_layers=2, hidden_size=10, expert_intermediate_size=5, shared_experts=1))
    with pytest.raises(ValueError, match="requires real expert_activation"):
        MoEComponent().estimate(desc, {"forward_mode": "prefill", "processed_tokens": 100})


def test_moe_cap_compatible_qwen_requires_real_activation():
    desc = ArchitectureDescriptor(
        model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        model_type="qwen2_moe",
        attention=AttentionDescriptor(type="mha", num_layers=2, hidden_size=16, num_attention_heads=2, num_key_value_heads=2, head_dim=8),
        cache=CacheDescriptor(type="kv", num_layers=2, head_dim=8, num_key_value_heads=2),
        moe=MoEDescriptor(enabled=True, moe_layers=2, routed_experts=8, top_k=2, hidden_size=16, expert_intermediate_size=4, shared_expert_intermediate_size=4),
        runtime=RuntimeDescriptor(precision_bytes=2, num_gpus=1, peak_bandwidth_tb=1, peak_flops_tf=100),
    )
    with pytest.raises(ValueError, match="requires real expert_activation"):
        estimate_moe_cap_compatible(desc, [{"forward_mode": "prefill", "latency": 1.0, "seq_lens_sum": 100, "batch_size": 1, "processed_tokens": 100}])
    explicit = estimate_moe_cap_compatible(desc, [{"forward_mode": "prefill", "latency": 1.0, "seq_lens_sum": 100, "batch_size": 1, "processed_tokens": 100, "expert_activation": 2, "raw_probe_source": "expert_distribution_metrics"}])
    assert explicit.prefill_smbu > 0
    assert explicit.prefill_smfu > 0


def test_moe_cap_qwen3_uses_dense_and_moe_layer_split():
    qwen3 = descriptor_from_config(
        {
            "model_type": "qwen3_moe",
            "num_hidden_layers": 4,
            "hidden_size": 16,
            "intermediate_size": 64,
            "moe_intermediate_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "mlp_only_layers": [0],
            "decoder_sparse_step": 2,
        },
        model_name="Qwen/Qwen3-30B-A3B",
        precision_bytes=2,
        num_gpus=1,
        peak_bandwidth_tb=1,
        peak_flops_tf=100,
    )
    qwen15_style = ArchitectureDescriptor(
        model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        model_type="qwen2_moe",
        attention=qwen3.attention,
        cache=qwen3.cache,
        ffn=qwen3.ffn,
        moe=MoEDescriptor(
            enabled=True,
            moe_layers=qwen3.attention.num_layers,
            routed_experts=8,
            top_k=2,
            hidden_size=16,
            expert_intermediate_size=8,
            shared_expert_intermediate_size=64,
        ),
        runtime=qwen3.runtime,
    )
    record = {"forward_mode": "prefill", "latency": 1.0, "seq_lens_sum": 100, "batch_size": 1, "processed_tokens": 100, "expert_activation": 4}
    qwen3_result = estimate_moe_cap_compatible(qwen3, [record])
    qwen15_result = estimate_moe_cap_compatible(qwen15_style, [record])
    assert qwen3_result.prefill_smbu > 0
    assert qwen3_result.prefill_smbu < qwen15_result.prefill_smbu


def qwen35_35b_config():
    return {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "hidden_size": 2048,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "mlp_only_layers": [],
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10,
        },
    }


def qwen36_35b_config():
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "full_attention_interval": 4,
            "num_hidden_layers": 40,
            "hidden_size": 2048,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "max_position_embeddings": 262144,
        },
        "vision_config": {
            "model_type": "qwen3_5_moe",
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
        },
    }


def test_qwen35_35b_descriptor_counts_hybrid_layers_from_text_config():
    desc = descriptor_from_config(qwen35_35b_config(), model_name="Qwen/Qwen3.5-35B-A3B")
    assert desc.model_type == "qwen3_5_moe_text"
    assert desc.attention.type == "hybrid"
    assert desc.cache.type == "hybrid"
    assert desc.attention.num_layers == 40
    assert desc.attention.full_attention_layers == 10
    assert desc.attention.linear_attention_layers == 30
    assert desc.moe.enabled is True
    assert desc.moe.moe_layers == 40
    assert desc.ffn.dense_layers == 0
    assert desc.moe.routed_experts == 256
    assert desc.moe.top_k == 8
    assert desc.moe.shared_expert_intermediate_size == 512


def test_qwen36_35b_descriptor_uses_text_config_not_vision_config():
    desc = descriptor_from_config(qwen36_35b_config(), model_name="Qwen/Qwen3.6-35B-A3B")
    assert desc.model_type == "qwen3_5_moe_text"
    assert desc.attention.type == "hybrid"
    assert desc.cache.type == "hybrid"
    assert desc.attention.num_layers == 40
    assert desc.attention.full_attention_layers == 10
    assert desc.attention.linear_attention_layers == 30
    assert desc.attention.layer_types == ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 10
    assert desc.attention.hidden_size == 2048
    assert desc.moe.enabled is True
    assert desc.moe.routed_experts == 256
    assert desc.moe.top_k == 8
    assert desc.moe.shared_expert_intermediate_size == 512


def test_qwen36_35b_adapter_uses_qwen_hybrid_component_mix():
    result = resolve_adapter(qwen36_35b_config(), model_name="Qwen/Qwen3.6-35B-A3B")
    assert result.name == "qwen_hybrid"
    assert result.descriptor.moe.enabled is True
    assert result.descriptor.attention.type == "hybrid"


def test_qwen35_hybrid_cache_uses_full_attention_layers_not_total_layers():
    desc = descriptor_from_config(qwen35_35b_config(), model_name="Qwen/Qwen3.5-35B-A3B")
    hybrid_units = CacheComponent().per_token_units(desc)
    old_all_kv_units = 2 * desc.attention.num_layers * desc.attention.num_key_value_heads * desc.attention.head_dim
    full_kv_units = 2 * desc.attention.full_attention_layers * desc.attention.num_key_value_heads * desc.attention.head_dim
    assert hybrid_units >= full_kv_units
    assert hybrid_units < old_all_kv_units


def test_qwen35_hybrid_attention_score_uses_full_attention_layers_not_total_layers():
    desc = descriptor_from_config(qwen35_35b_config(), model_name="Qwen/Qwen3.5-35B-A3B")
    record = {"forward_mode": "decode", "seq_lens_sum": 1000, "batch_size": 4, "expert_activation": 8}
    score = AttentionComponent().attention_score_units(desc, record)
    old_dense_only_score = 1000 * desc.attention.num_layers * desc.attention.num_attention_heads * desc.attention.head_dim * 2 / 4 / 1e12
    assert score < old_dense_only_score
    assert score > 0


def test_qwen35_moe_cap_uses_hybrid_formula_and_shared_expert():
    desc = descriptor_from_config(
        qwen35_35b_config(),
        model_name="Qwen/Qwen3.5-35B-A3B",
        precision_bytes=2,
        num_gpus=1,
        peak_bandwidth_tb=1,
        peak_flops_tf=100,
    )
    record = {
        "forward_mode": "prefill",
        "latency": 1.0,
        "seq_lens_sum": 100,
        "batch_size": 1,
        "processed_tokens": 100,
        "expert_activation": 8,
        "raw_probe_source": "expert_distribution_metrics",
    }
    result = estimate_moe_cap_compatible(desc, [record])
    assert result.prefill_smbu > 0
    assert result.prefill_smfu > 0


def test_qwen36_moe_cap_uses_hybrid_formula_and_real_expert_activation():
    desc = descriptor_from_config(
        qwen36_35b_config(),
        model_name="Qwen/Qwen3.6-35B-A3B",
        precision_bytes=2,
        num_gpus=4,
        peak_bandwidth_tb=2,
        peak_flops_tf=312,
    )
    record = {
        "forward_mode": "decode",
        "latency": 0.01,
        "seq_lens_sum": 1000,
        "batch_size": 8,
        "processed_tokens": 8,
        "expert_activation": 8,
        "raw_probe_source": "expert_distribution_metrics",
    }
    result = estimate_moe_cap_compatible(desc, [record])
    assert result.decoding_smbu > 0
    assert result.decoding_smfu > 0


def test_qwen36_moe_cap_validates_complete_hybrid_descriptor():
    desc = ArchitectureDescriptor(
        model_name="Qwen/Qwen3.6-35B-A3B",
        model_type="qwen3_5_moe_text",
        attention=AttentionDescriptor(type="gqa", num_layers=40, hidden_size=2048, num_attention_heads=16, num_key_value_heads=2, head_dim=256),
        moe=MoEDescriptor(enabled=True, moe_layers=40, routed_experts=256, top_k=8, hidden_size=2048, expert_intermediate_size=512, shared_expert_intermediate_size=512),
    )
    status = support_status(desc)
    assert not status.supported
    assert "requires hybrid attention" in status.reason


def test_qwen36_moe_cap_does_not_use_top_k_or_router_for_smfu():
    desc = descriptor_from_config(
        qwen36_35b_config(),
        model_name="Qwen/Qwen3.6-35B-A3B",
        precision_bytes=2,
        num_gpus=1,
        peak_bandwidth_tb=1,
        peak_flops_tf=100,
    )
    record = {
        "forward_mode": "decode",
        "latency": 1.0,
        "seq_lens_sum": 1000,
        "batch_size": 1,
        "processed_tokens": 1,
        "expert_activation": 8,
        "raw_probe_source": "expert_distribution_metrics",
    }
    baseline = estimate_moe_cap_compatible(desc, [record]).decoding_smfu
    changed_topk = descriptor_from_config(
        {**qwen36_35b_config(), "text_config": {**qwen36_35b_config()["text_config"], "num_experts_per_tok": 16}},
        model_name="Qwen/Qwen3.6-35B-A3B",
        precision_bytes=2,
        num_gpus=1,
        peak_bandwidth_tb=1,
        peak_flops_tf=100,
    )
    changed_router = descriptor_from_config(
        {**qwen36_35b_config(), "text_config": {**qwen36_35b_config()["text_config"], "num_experts": 512}},
        model_name="Qwen/Qwen3.6-35B-A3B",
        precision_bytes=2,
        num_gpus=1,
        peak_bandwidth_tb=1,
        peak_flops_tf=100,
    )
    assert estimate_moe_cap_compatible(changed_topk, [record]).decoding_smfu == baseline
    assert estimate_moe_cap_compatible(changed_router, [record]).decoding_smfu == baseline
