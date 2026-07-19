"""Candidate-action policy and categorical value network."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .features import (
    HISTORY_FEATURE_SIZE,
    OPTION_FEATURE_SIZE,
    SEMANTIC_FEATURE_SIZE,
    STATE_FEATURE_SIZE,
    TOKEN_FEATURE_SIZE,
)


class CandidatePolicyValueNet(nn.Module):
    """Score each currently legal option and predict a categorical return."""

    def __init__(self, hidden_size: int = 128, value_bins: int = 101) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.value_bins = value_bins
        self.select_embedding = nn.Embedding(16, 8)
        self.context_embedding = nn.Embedding(64, 16)
        self.option_type_embedding = nn.Embedding(32, 8)
        self.area_embedding = nn.Embedding(16, 4)
        self.in_play_area_embedding = nn.Embedding(16, 4)

        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_FEATURE_SIZE + 24, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.option_encoder = nn.Sequential(
            nn.Linear(OPTION_FEATURE_SIZE + 16, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, value_bins),
        )
        self.register_buffer("value_support", torch.linspace(-1.0, 1.0, value_bins))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        select_ids = batch["select_type"].clamp(0, self.select_embedding.num_embeddings - 1)
        context_ids = batch["context"].clamp(0, self.context_embedding.num_embeddings - 1)
        state_input = torch.cat(
            [
                batch["state"],
                self.select_embedding(select_ids),
                self.context_embedding(context_ids),
            ],
            dim=-1,
        )
        state_hidden = self.state_encoder(state_input)

        option_types = batch["option_types"].clamp(0, self.option_type_embedding.num_embeddings - 1)
        areas = batch["areas"].clamp(0, self.area_embedding.num_embeddings - 1)
        in_play_areas = batch["in_play_areas"].clamp(
            0, self.in_play_area_embedding.num_embeddings - 1
        )
        option_input = torch.cat(
            [
                batch["options"],
                self.option_type_embedding(option_types),
                self.area_embedding(areas),
                self.in_play_area_embedding(in_play_areas),
            ],
            dim=-1,
        )
        option_hidden = self.option_encoder(option_input)
        expanded_state = state_hidden.unsqueeze(1).expand_as(option_hidden)
        policy_input = torch.cat(
            [expanded_state, option_hidden, expanded_state * option_hidden], dim=-1
        )
        policy_logits = self.policy_head(policy_input).squeeze(-1)
        policy_logits = policy_logits.masked_fill(~batch["option_mask"], torch.finfo().min)
        return policy_logits, self.value_head(state_hidden)

    def expected_value(self, value_logits: Tensor) -> Tensor:
        return (value_logits.softmax(dim=-1) * self.value_support).sum(dim=-1)


class TokenPolicyValueNet(nn.Module):
    """Attend over visible card tokens, then score every legal candidate action."""

    def __init__(
        self,
        hidden_size: int = 256,
        value_bins: int = 101,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        card_vocab_size: int = 2048,
        attack_vocab_size: int = 2048,
        semantic_feature_size: int = 0,
        history_feature_size: int = 0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.value_bins = value_bins
        self.card_vocab_size = card_vocab_size
        self.attack_vocab_size = attack_vocab_size
        self.semantic_feature_size = semantic_feature_size
        self.history_feature_size = history_feature_size
        if semantic_feature_size not in {0, SEMANTIC_FEATURE_SIZE}:
            raise ValueError("semantic_feature_size does not match the V3 schema")
        if history_feature_size not in {0, HISTORY_FEATURE_SIZE}:
            raise ValueError("history_feature_size does not match the V3 schema")

        self.select_embedding = nn.Embedding(16, hidden_size)
        self.context_embedding = nn.Embedding(64, hidden_size)
        self.card_embedding = nn.Embedding(card_vocab_size, hidden_size, padding_idx=0)
        self.attack_embedding = nn.Embedding(attack_vocab_size, hidden_size, padding_idx=0)
        self.token_kind_embedding = nn.Embedding(8, hidden_size)
        self.zone_embedding = nn.Embedding(16, hidden_size)
        self.owner_embedding = nn.Embedding(3, hidden_size)
        self.slot_embedding = nn.Embedding(16, hidden_size)
        self.card_type_embedding = nn.Embedding(8, hidden_size)
        self.energy_type_embedding = nn.Embedding(16, hidden_size)
        self.option_type_embedding = nn.Embedding(32, hidden_size)
        self.area_embedding = nn.Embedding(16, hidden_size)
        self.special_condition_embedding = nn.Embedding(16, hidden_size)
        if semantic_feature_size:
            self.token_semantic_projection: nn.Linear | None = nn.Linear(
                semantic_feature_size, hidden_size
            )
            self.option_semantic_projection: nn.Linear | None = nn.Linear(
                semantic_feature_size, hidden_size
            )
        else:
            self.token_semantic_projection = None
            self.option_semantic_projection = None
        if history_feature_size:
            self.history_projection: nn.Linear | None = nn.Linear(
                history_feature_size, hidden_size
            )
            self.history_type_embedding: nn.Embedding | None = nn.Embedding(32, hidden_size)
        else:
            self.history_projection = None
            self.history_type_embedding = None

        self.state_projection = nn.Linear(STATE_FEATURE_SIZE, hidden_size)
        self.token_projection = nn.Linear(TOKEN_FEATURE_SIZE, hidden_size)
        self.option_projection = nn.Linear(OPTION_FEATURE_SIZE, hidden_size)
        self.state_norm = nn.LayerNorm(hidden_size)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.option_norm = nn.LayerNorm(hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        # Nested-tensor mask conversion is not implemented on MPS as of PyTorch 2.13.
        self.state_transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.option_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, value_bins),
        )
        self.register_buffer("value_support", torch.linspace(-1.0, 1.0, value_bins))

    @staticmethod
    def _embedding_ids(values: Tensor, size: int) -> Tensor:
        return torch.where((values >= 0) & (values < size), values, torch.zeros_like(values))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        select_ids = batch["select_type"].clamp(0, self.select_embedding.num_embeddings - 1)
        context_ids = batch["context"].clamp(0, self.context_embedding.num_embeddings - 1)
        state_token = self.state_norm(
            self.state_projection(batch["state"])
            + self.select_embedding(select_ids)
            + self.context_embedding(context_ids)
        )

        card_ids = self._embedding_ids(batch["token_card_ids"], self.card_vocab_size)
        token_hidden = (
            self.token_projection(batch["tokens"])
            + self.card_embedding(card_ids)
            + self.token_kind_embedding(batch["token_kinds"].clamp(0, 7))
            + self.zone_embedding(batch["token_zones"].clamp(0, 15))
            + self.owner_embedding(batch["token_owners"].clamp(0, 2))
            + self.slot_embedding(batch["token_slots"].clamp(0, 15))
            + self.card_type_embedding(batch["token_card_types"].clamp(0, 7))
            + self.energy_type_embedding(batch["token_energy_types"].clamp(0, 15))
            + self.energy_type_embedding(batch["token_weaknesses"].clamp(0, 15))
            + self.energy_type_embedding(batch["token_resistances"].clamp(0, 15))
        )
        if self.token_semantic_projection is not None:
            token_hidden = token_hidden + self.token_semantic_projection(
                batch["token_semantics"]
            )
        token_hidden = self.token_norm(token_hidden)
        sequence_parts = [state_token.unsqueeze(1), token_hidden]
        mask_parts = [
            torch.ones(state_token.shape[0], 1, dtype=torch.bool, device=state_token.device),
            batch["token_mask"],
        ]
        if self.history_projection is not None and self.history_type_embedding is not None:
            history_card_ids = self._embedding_ids(
                batch["history_card_ids"], self.card_vocab_size
            )
            history_target_card_ids = self._embedding_ids(
                batch["history_target_card_ids"], self.card_vocab_size
            )
            history_attack_ids = self._embedding_ids(
                batch["history_attack_ids"], self.attack_vocab_size
            )
            history_hidden = (
                self.history_projection(batch["history_features"])
                + self.history_type_embedding(batch["history_types"].clamp(0, 31))
                + self.owner_embedding(batch["history_owners"].clamp(0, 2))
                + self.card_embedding(history_card_ids)
                + self.card_embedding(history_target_card_ids)
                + self.attack_embedding(history_attack_ids)
                + self.zone_embedding(batch["history_from_zones"].clamp(0, 15))
                + self.zone_embedding(batch["history_to_zones"].clamp(0, 15))
            )
            sequence_parts.append(self.token_norm(history_hidden))
            mask_parts.append(batch["history_mask"])
        state_sequence = torch.cat(sequence_parts, dim=1)
        state_mask = torch.cat(mask_parts, dim=1)
        encoded_state = self.state_transformer(
            state_sequence,
            src_key_padding_mask=~state_mask,
        )
        pooled_state = encoded_state[:, 0]

        option_card_ids = self._embedding_ids(batch["option_card_ids"], self.card_vocab_size)
        option_attack_ids = self._embedding_ids(
            batch["option_attack_ids"], self.attack_vocab_size
        )
        option_hidden = (
            self.option_projection(batch["options"])
            + self.card_embedding(option_card_ids)
            + self.attack_embedding(option_attack_ids)
            + self.option_type_embedding(batch["option_types"].clamp(0, 31))
            + self.area_embedding(batch["areas"].clamp(0, 15))
            + self.area_embedding(batch["in_play_areas"].clamp(0, 15))
            + self.special_condition_embedding(
                batch["option_special_conditions"].clamp(0, 15)
            )
        )
        if self.option_semantic_projection is not None:
            option_hidden = option_hidden + self.option_semantic_projection(
                batch["option_semantics"]
            )
        option_hidden = self.option_norm(option_hidden)
        attended_options, _ = self.option_attention(
            option_hidden,
            encoded_state,
            encoded_state,
            key_padding_mask=~state_mask,
            need_weights=False,
        )
        option_hidden = option_hidden + attended_options
        expanded_state = pooled_state.unsqueeze(1).expand_as(option_hidden)
        policy_input = torch.cat(
            [expanded_state, option_hidden, expanded_state * option_hidden], dim=-1
        )
        policy_logits = self.policy_head(policy_input).squeeze(-1)
        policy_logits = policy_logits.masked_fill(~batch["option_mask"], torch.finfo().min)

        option_weights = batch["option_mask"].unsqueeze(-1).to(option_hidden.dtype)
        pooled_options = (option_hidden * option_weights).sum(dim=1) / option_weights.sum(
            dim=1
        ).clamp_min(1.0)
        value_logits = self.value_head(torch.cat([pooled_state, pooled_options], dim=-1))
        return policy_logits, value_logits

    def expected_value(self, value_logits: Tensor) -> Tensor:
        return (value_logits.softmax(dim=-1) * self.value_support).sum(dim=-1)


PolicyValueModel = CandidatePolicyValueNet | TokenPolicyValueNet


def build_model(config: dict) -> PolicyValueModel:
    """Build old and new checkpoints through one backward-compatible factory."""
    values = dict(config)
    model_type = values.pop("model_type", "mlp_v1")
    if model_type == "mlp_v1":
        return CandidatePolicyValueNet(**values)
    if model_type == "transformer_v2":
        return TokenPolicyValueNet(**values)
    if model_type == "transformer_v3":
        use_card_semantics = bool(values.pop("use_card_semantics", True))
        use_history = bool(values.pop("use_history", True))
        values.setdefault(
            "semantic_feature_size", SEMANTIC_FEATURE_SIZE if use_card_semantics else 0
        )
        values.setdefault("history_feature_size", HISTORY_FEATURE_SIZE if use_history else 0)
        return TokenPolicyValueNet(**values)
    raise ValueError(f"Unsupported model_type: {model_type}")


def encoder_version(config: dict) -> int:
    return {
        "transformer_v2": 2,
        "transformer_v3": 3,
    }.get(config.get("model_type"), 1)


def categorical_value_targets(
    values: Tensor,
    support: Tensor,
    sigma: float = 0.05,
) -> Tensor:
    """Create smooth histogram targets similar to HL-Gauss value classification."""
    distances = values.unsqueeze(-1) - support.unsqueeze(0)
    targets = torch.exp(-0.5 * (distances / sigma).square())
    return targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
