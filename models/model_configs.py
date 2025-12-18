"""
Standard Model Configurations for NeuroMap.

Provides reproducible configurations for different model architectures
used in modular arithmetic experiments.

These configurations are designed to match those used in:
- Neel Nanda's "Progress Measures for Grokking"
- TransformerLens standard configurations
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ModelConfig:
    """Base configuration for all models."""
    name: str = "base"
    vocab_size: int = 17  # Modulus p
    d_model: int = 64
    n_ctx: int = 2  # Context length (a, b)
    device: str = "cpu"
    seed: int = 42


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for transformer models."""
    name: str = "transformer"
    n_layers: int = 2
    n_heads: int = 4
    d_head: int = 16
    d_mlp: int = 256
    act_fn: str = "relu"
    normalization_type: str = "LN"
    use_attn_scale: bool = True
    attn_only: bool = False  # If True, no MLP layers


@dataclass
class MambaConfig(ModelConfig):
    """Configuration for Mamba/SSM models."""
    name: str = "mamba"
    n_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2  # MLP expansion factor


@dataclass
class MemoryConfig(ModelConfig):
    """Configuration for memory-based models."""
    name: str = "memory"
    memory_type: str = "direct_lookup"  # "direct_lookup" or "hybrid"
    hidden_dim: int = 64
    use_gating: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

    # Training
    epochs: int = 10000
    batch_size: int = "full"  # "full" or integer
    gradient_clip: float = 1.0

    # Scheduler
    scheduler: str = "none"  # "none", "cosine", "linear"
    warmup_steps: int = 0

    # Logging
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 1000


# Pre-defined configurations for different experiments
# These match configurations from mechanistic interpretability literature

STANDARD_2L_TRANSFORMER = TransformerConfig(
    name="standard_2l_transformer",
    n_layers=2,
    d_model=64,
    n_heads=4,
    d_head=16,
    d_mlp=256,
    act_fn="relu",
    normalization_type="LN",
)

GROKKING_TRANSFORMER = TransformerConfig(
    name="grokking_transformer",
    n_layers=1,
    d_model=128,
    n_heads=4,
    d_head=32,
    d_mlp=512,
    act_fn="gelu",
    normalization_type="LN",
    use_attn_scale=True,
)

TINY_TRANSFORMER = TransformerConfig(
    name="tiny_transformer",
    n_layers=1,
    d_model=32,
    n_heads=2,
    d_head=16,
    d_mlp=64,
    act_fn="relu",
    normalization_type="LN",
)

ATTN_ONLY_TRANSFORMER = TransformerConfig(
    name="attn_only_transformer",
    n_layers=2,
    d_model=64,
    n_heads=4,
    d_head=16,
    d_mlp=0,  # No MLP
    act_fn="relu",
    normalization_type="LN",
    attn_only=True,
)

STANDARD_MAMBA = MambaConfig(
    name="standard_mamba",
    n_layers=2,
    d_model=64,
    d_state=16,
    d_conv=4,
    expand=2,
)

DIRECT_LOOKUP_MEMORY = MemoryConfig(
    name="direct_lookup",
    memory_type="direct_lookup",
    d_model=64,
)

HYBRID_MEMORY = MemoryConfig(
    name="hybrid_memory",
    memory_type="hybrid",
    hidden_dim=64,
    use_gating=True,
)

# Training configurations for different scenarios

STANDARD_TRAINING = TrainingConfig(
    optimizer="adamw",
    learning_rate=1e-3,
    weight_decay=0.01,
    epochs=10000,
    batch_size="full",
)

GROKKING_TRAINING = TrainingConfig(
    optimizer="adamw",
    learning_rate=1e-3,
    weight_decay=1.0,  # High weight decay for grokking
    epochs=50000,
    batch_size="full",
)

QUICK_TRAINING = TrainingConfig(
    optimizer="adam",
    learning_rate=1e-2,
    weight_decay=0.0,
    epochs=1000,
    batch_size="full",
)

MEMORY_TRAINING = TrainingConfig(
    optimizer="adam",
    learning_rate=1e-2,
    weight_decay=0.0,
    epochs=100,
    batch_size="full",
)


# Configurations indexed by modulus p
CONFIGS_BY_MODULUS = {
    5: {
        "transformer": TransformerConfig(vocab_size=5, d_model=32, n_heads=2, d_mlp=64),
        "memory": MemoryConfig(vocab_size=5, hidden_dim=32),
    },
    7: {
        "transformer": TransformerConfig(vocab_size=7, d_model=64, n_heads=4, d_mlp=128),
        "memory": MemoryConfig(vocab_size=7, hidden_dim=64),
    },
    13: {
        "transformer": TransformerConfig(vocab_size=13, d_model=64, n_heads=4, d_mlp=256),
        "memory": MemoryConfig(vocab_size=13, hidden_dim=64),
    },
    17: {
        "transformer": STANDARD_2L_TRANSFORMER,
        "memory": MemoryConfig(vocab_size=17, hidden_dim=64),
    },
    23: {
        "transformer": TransformerConfig(vocab_size=23, d_model=128, n_heads=4, d_mlp=512),
        "memory": MemoryConfig(vocab_size=23, hidden_dim=128),
    },
}


def get_config(model_type: str, modulus: int = 17) -> ModelConfig:
    """
    Get configuration for a model type and modulus.

    Args:
        model_type: Type of model ("transformer", "mamba", "memory")
        modulus: Modulus p for the modular arithmetic task

    Returns:
        Appropriate ModelConfig
    """
    if modulus in CONFIGS_BY_MODULUS:
        configs = CONFIGS_BY_MODULUS[modulus]
        if model_type in configs:
            return configs[model_type]

    # Default configurations
    if model_type == "transformer":
        return TransformerConfig(vocab_size=modulus)
    elif model_type == "mamba":
        return MambaConfig(vocab_size=modulus)
    elif model_type == "memory":
        return MemoryConfig(vocab_size=modulus)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_training_config(scenario: str = "standard") -> TrainingConfig:
    """
    Get training configuration for a scenario.

    Args:
        scenario: Training scenario ("standard", "grokking", "quick", "memory")

    Returns:
        TrainingConfig
    """
    configs = {
        "standard": STANDARD_TRAINING,
        "grokking": GROKKING_TRAINING,
        "quick": QUICK_TRAINING,
        "memory": MEMORY_TRAINING,
    }

    if scenario in configs:
        return configs[scenario]

    raise ValueError(f"Unknown training scenario: {scenario}")


def config_to_dict(config: ModelConfig) -> Dict[str, Any]:
    """Convert a config dataclass to dictionary."""
    return {k: v for k, v in config.__dict__.items()}


def dict_to_config(d: Dict[str, Any], config_class=TransformerConfig) -> ModelConfig:
    """Convert dictionary to config dataclass."""
    return config_class(**d)


# Export all configurations
ALL_TRANSFORMER_CONFIGS = [
    STANDARD_2L_TRANSFORMER,
    GROKKING_TRANSFORMER,
    TINY_TRANSFORMER,
    ATTN_ONLY_TRANSFORMER,
]

ALL_MAMBA_CONFIGS = [
    STANDARD_MAMBA,
]

ALL_MEMORY_CONFIGS = [
    DIRECT_LOOKUP_MEMORY,
    HYBRID_MEMORY,
]


if __name__ == "__main__":
    print("NeuroMap Model Configurations")
    print("=" * 50)

    print("\nTransformer Configurations:")
    for config in ALL_TRANSFORMER_CONFIGS:
        print(f"  {config.name}: {config.n_layers}L, {config.d_model}D, {config.n_heads}H")

    print("\nMamba Configurations:")
    for config in ALL_MAMBA_CONFIGS:
        print(f"  {config.name}: {config.n_layers}L, {config.d_model}D")

    print("\nMemory Configurations:")
    for config in ALL_MEMORY_CONFIGS:
        print(f"  {config.name}: {config.memory_type}")

    print("\nConfigurations by modulus:")
    for p, configs in CONFIGS_BY_MODULUS.items():
        print(f"  p={p}: {list(configs.keys())}")

    print("\nModel configurations complete!")
