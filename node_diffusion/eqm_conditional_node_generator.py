import contextlib
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from .conditional_denoising_node_generator import (
    CrossTransformerEncoderLayer,
    EdgeMLP,
    GraphWithEdgesDataset,
    MetricsLogger,
    collate_graph_with_edges,
    plot_metrics,
    suppress_output,
)
from .conditional_node_generator_base import ConditionalNodeGeneratorBase


class EqMConditionalNodeGeneratorModule(pl.LightningModule):
    """Conditional EqM model with an explicit scalar energy and score via autograd."""

    def __init__(
        self,
        number_of_rows_per_example: int,
        input_feature_dimension: int,
        condition_feature_dimension: int,
        latent_embedding_dimension: int,
        number_of_transformer_layers: int,
        transformer_attention_head_count: int,
        transformer_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        verbose: bool = False,
        important_feature_index: int = 1,
        max_degree: Optional[int] = None,
        lambda_degree_importance: float = 1.0,
        degree_temperature: Optional[float] = None,
        degree_min_val: float = 0.0,
        degree_range_val: float = 1.0,
        lambda_node_exist_importance: float = 1.0,
        use_locality_supervision: bool = False,
        lambda_locality_importance: float = 1.0,
        exist_pos_weight: Union[torch.Tensor, float] = 1.0,
        eqm_sigma: float = 0.2,
        sampling_step_size: float = 0.05,
        sampling_steps: int = 100,
        langevin_noise_scale: float = 0.0,
        pool_condition_tokens: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["verbose"])

        if max_degree is None:
            raise ValueError("max_degree must be provided when initializing the EqM model.")
        if eqm_sigma <= 0:
            raise ValueError(f"eqm_sigma must be positive (got {eqm_sigma}).")
        if sampling_step_size <= 0:
            raise ValueError(f"sampling_step_size must be positive (got {sampling_step_size}).")
        if sampling_steps <= 0:
            raise ValueError(f"sampling_steps must be positive (got {sampling_steps}).")

        self.number_of_rows_per_example = number_of_rows_per_example
        self.input_feature_dimension = input_feature_dimension
        self.condition_feature_dimension = condition_feature_dimension
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.max_degree = int(max_degree)
        self.lambda_degree_importance = lambda_degree_importance
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.use_locality_supervision = bool(use_locality_supervision)
        self.lambda_locality_importance = lambda_locality_importance
        self.eqm_sigma = float(eqm_sigma)
        self.sampling_step_size = float(sampling_step_size)
        self.sampling_steps = int(sampling_steps)
        self.langevin_noise_scale = float(langevin_noise_scale)
        self.pool_condition_tokens = bool(pool_condition_tokens)
        self.use_guidance = False
        self.noise_degree_factor = 1.0

        self.register_buffer(
            "exist_pos_weight",
            torch.as_tensor(exist_pos_weight, dtype=torch.float32),
        )
        self.register_buffer("deg_min_val", torch.tensor(degree_min_val, dtype=torch.float32))
        self.register_buffer("deg_range_val", torch.tensor(max(degree_range_val, 1e-8), dtype=torch.float32))

        self.train_losses = []
        self.val_losses = []
        self.train_deg_ce = []
        self.val_deg_ce = []
        self.train_loss_all = []
        self.val_loss_all = []
        self.train_exist = []
        self.val_exist = []
        self.train_recon = []
        self.val_recon = []
        if self.use_locality_supervision:
            self.train_edge_loss = []
            self.val_edge_loss = []
            self.train_edge_acc = []
            self.val_edge_acc = []

        self.layernorm_in = nn.LayerNorm(input_feature_dimension, elementwise_affine=True)
        self.linear_encoder_input_to_latent = nn.Linear(input_feature_dimension, latent_embedding_dimension)
        self.linear_encoder_condition_to_latent = nn.Linear(condition_feature_dimension, latent_embedding_dimension)
        self.shared_transformer = nn.ModuleList(
            [
                CrossTransformerEncoderLayer(
                    embed_dim=latent_embedding_dimension,
                    num_heads=transformer_attention_head_count,
                    dropout=transformer_dropout,
                )
                for _ in range(number_of_transformer_layers)
            ]
        )
        self.potential_head = nn.Sequential(
            nn.LayerNorm(latent_embedding_dimension),
            nn.Linear(latent_embedding_dimension, latent_embedding_dimension),
            nn.GELU(),
            nn.Linear(latent_embedding_dimension, 1),
        )
        self.degree_head = nn.Linear(latent_embedding_dimension, self.max_degree + 1)
        self.exist_head = nn.Linear(latent_embedding_dimension, 1)
        if self.use_locality_supervision:
            self.edge_head = EdgeMLP(
                latent_dim=latent_embedding_dimension,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout,
            )

    def _encode_with_condition(
        self,
        input_rows: torch.Tensor,
        global_condition_vector: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_mask is not None:
            input_rows = input_rows * node_mask.unsqueeze(-1).to(dtype=input_rows.dtype)

        x_norm = self.layernorm_in(input_rows)
        latent_tokens = self.linear_encoder_input_to_latent(x_norm)

        if global_condition_vector.dim() == 2:
            cond_tokens = global_condition_vector.unsqueeze(1)
        elif global_condition_vector.dim() == 3:
            cond_tokens = global_condition_vector
        else:
            raise ValueError(
                "global_condition_vector must have shape (B, C) or (B, M, C); "
                f"received {tuple(global_condition_vector.shape)}"
            )

        if self.pool_condition_tokens and cond_tokens.size(1) > 1:
            cond_tokens = cond_tokens.mean(dim=1, keepdim=True)

        weight = self.linear_encoder_condition_to_latent.weight.transpose(0, 1)
        cond_proj = torch.matmul(cond_tokens, weight)
        bias = self.linear_encoder_condition_to_latent.bias
        if bias is not None:
            cond_proj = cond_proj + bias

        if node_mask is not None:
            latent_tokens = latent_tokens * node_mask.unsqueeze(-1).to(dtype=latent_tokens.dtype)

        self_padding_mask = None if node_mask is None else ~node_mask
        for layer in self.shared_transformer:
            latent_tokens = layer(
                latent_tokens,
                k=cond_proj,
                v=cond_proj,
                self_key_padding_mask=self_padding_mask,
                query_mask=node_mask,
            )
            if node_mask is not None:
                latent_tokens = latent_tokens * node_mask.unsqueeze(-1).to(dtype=latent_tokens.dtype)
        return latent_tokens

    def _build_noise_scale(self, x: torch.Tensor) -> torch.Tensor:
        noise_scale = torch.full_like(x, self.eqm_sigma)
        noise_scale[..., self.important_feature_index] /= self.noise_degree_factor
        return noise_scale

    def _compute_score_field(
        self,
        noisy_input: torch.Tensor,
        global_condition_vector: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        *,
        create_graph: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_tokens = self._encode_with_condition(noisy_input, global_condition_vector, node_mask=node_mask)
        phi_per_node = self.potential_head(latent_tokens).squeeze(-1)
        if node_mask is not None:
            phi_per_node = phi_per_node * node_mask.to(dtype=phi_per_node.dtype)
        phi = phi_per_node.sum(dim=1)
        score = -torch.autograd.grad(
            phi.sum(),
            noisy_input,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        return score, phi, latent_tokens

    def _eqm_loss(
        self,
        input_examples: torch.Tensor,
        global_condition: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        *,
        create_graph: bool,
    ) -> Tuple[dict, torch.Tensor]:
        eps = torch.randn_like(input_examples)
        noise_scale = self._build_noise_scale(input_examples)
        noisy_input = (input_examples + noise_scale * eps).detach().requires_grad_(True)
        score_mask = (
            node_mask.unsqueeze(-1).to(dtype=input_examples.dtype)
            if node_mask is not None
            else torch.ones_like(input_examples[..., :1])
        )

        score, _, latent_noisy = self._compute_score_field(
            noisy_input,
            global_condition,
            node_mask=node_mask,
            create_graph=create_graph,
        )
        target_score = -eps / noise_scale
        eqm_error = (score - target_score).pow(2) * score_mask
        loss_eqm = eqm_error.sum() / score_mask.sum().clamp_min(1.0)

        denoised = noisy_input + noise_scale.pow(2) * score
        latent_clean = self._encode_with_condition(denoised, global_condition, node_mask=node_mask)
        logits_deg = self.degree_head(latent_clean)
        logits_exist = self.exist_head(latent_clean).squeeze(-1)

        target_exist = (input_examples[..., 0] >= 0.5).float()
        exist_loss_map = F.binary_cross_entropy_with_logits(
            logits_exist,
            target_exist,
            pos_weight=self.exist_pos_weight,
            reduction="none",
        )
        exist_mask = node_mask.to(dtype=exist_loss_map.dtype) if node_mask is not None else torch.ones_like(exist_loss_map)
        loss_exist = (exist_loss_map * exist_mask).sum() / exist_mask.sum().clamp_min(1.0)

        target_degree_scaled = input_examples[..., self.important_feature_index]
        deg_unscaled = target_degree_scaled * self.deg_range_val + self.deg_min_val
        true_deg_class = torch.clamp(torch.round(deg_unscaled), 0, self.max_degree).long()

        if self.degree_temperature is not None and self.degree_temperature > 0:
            logits_for_loss = logits_deg / self.degree_temperature
        else:
            logits_for_loss = logits_deg

        deg_loss_map = F.cross_entropy(
            logits_for_loss.reshape(-1, self.max_degree + 1),
            true_deg_class.reshape(-1),
            reduction="none",
        )
        deg_mask = exist_mask.reshape(-1)
        loss_deg_ce = (deg_loss_map * deg_mask).sum() / deg_mask.sum().clamp_min(1.0)

        total_loss = (
            loss_eqm
            + self.lambda_node_exist_importance * loss_exist
            + self.lambda_degree_importance * loss_deg_ce
        )
        return (
            {
                "total": total_loss,
                "eqm": loss_eqm,
                "exist": loss_exist,
                "deg_ce": loss_deg_ce,
            },
            latent_clean if self.use_locality_supervision else latent_noisy,
        )

    def training_step(self, batch, batch_idx):
        if self.use_locality_supervision:
            input_examples, global_condition, edge_idx, edge_labels, node_mask = batch
        else:
            input_examples, global_condition, node_mask = batch

        losses, latent_tokens = self._eqm_loss(
            input_examples,
            global_condition,
            node_mask=node_mask,
            create_graph=True,
        )
        total_loss = losses["total"]

        if self.use_locality_supervision and edge_idx.numel() > 0:
            b, i, j = edge_idx.unbind(1)
            h_i = latent_tokens[b, i]
            h_j = latent_tokens[b, j]
            edge_logits = self.edge_head(h_i, h_j)
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
            total_loss = total_loss + self.lambda_locality_importance * edge_loss

            with torch.no_grad():
                edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
                edge_acc = (edge_pred == edge_labels).float().mean()

            self.log("train_edge_loss", edge_loss, on_step=False, on_epoch=True)
            self.log("train_edge_acc", edge_acc, on_step=False, on_epoch=True)

        self.log("train_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon", losses["eqm"], on_step=False, on_epoch=True)
        self.log("train_deg_ce", losses["deg_ce"], on_step=False, on_epoch=True)
        self.log("train_exist", losses["exist"], on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            if self.use_locality_supervision:
                input_examples, global_condition, edge_idx, edge_labels, node_mask = batch
            else:
                input_examples, global_condition, node_mask = batch

            losses, latent_tokens = self._eqm_loss(
                input_examples,
                global_condition,
                node_mask=node_mask,
                create_graph=False,
            )
            total_loss = losses["total"]

            if self.use_locality_supervision and edge_idx.numel() > 0:
                b, i, j = edge_idx.unbind(1)
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                edge_logits = self.edge_head(h_i, h_j)
                edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
                total_loss = total_loss + self.lambda_locality_importance * edge_loss

                edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
                edge_acc = (edge_pred == edge_labels).float().mean()

                self.log("val_edge_loss", edge_loss, on_step=False, on_epoch=True)
                self.log("val_edge_acc", edge_acc, on_step=False, on_epoch=True)

            self.log("val_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_recon", losses["eqm"], on_step=False, on_epoch=True)
            self.log("val_deg_ce", losses["deg_ce"], on_step=False, on_epoch=True)
            self.log("val_exist", losses["exist"], on_step=False, on_epoch=True)
        return total_loss.detach()

    def on_train_end(self):
        if not self.verbose:
            return
        plot_metrics(
            train_metrics={
                "total": self.train_losses,
                "deg_ce": self.train_deg_ce,
                "eqm": self.train_recon,
                "exist": self.train_exist,
                **({"locality": self.train_edge_loss} if self.use_locality_supervision else {}),
            },
            val_metrics={
                "total": self.val_losses,
                "deg_ce": self.val_deg_ce,
                "eqm": self.val_recon,
                "exist": self.val_exist,
                **({"locality": self.val_edge_loss} if self.use_locality_supervision else {}),
            },
            window=10,
            alpha=0.1,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def set_guidance_classifier(self, num_classes: int) -> None:
        raise NotImplementedError("Classifier guidance is not implemented for EqMConditionalNodeGenerator phase 1.")

    def train_guidance_classifier(self, *args, **kwargs):
        raise NotImplementedError("Classifier guidance is not implemented for EqMConditionalNodeGenerator phase 1.")

    def generate(
        self,
        global_condition: torch.Tensor,
        total_steps: Optional[int] = None,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
        use_heads_projection: bool = False,
        exist_threshold: float = 0.5,
    ) -> torch.Tensor:
        del desired_class
        self.eval()
        steps = int(total_steps if total_steps is not None else self.sampling_steps)
        eta = self.sampling_step_size
        batch_size = global_condition.size(0)
        device = global_condition.device

        node_mask = None
        x = torch.randn(
            batch_size,
            self.number_of_rows_per_example,
            self.input_feature_dimension,
            device=device,
        )

        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            with torch.enable_grad():
                score, _, latent_tokens = self._compute_score_field(
                    x,
                    global_condition,
                    node_mask=node_mask,
                    create_graph=False,
                )
            x = (x + eta * score).detach()
            if self.langevin_noise_scale > 0:
                x = x + np.sqrt(2.0 * eta) * self.langevin_noise_scale * torch.randn_like(x)

        if use_heads_projection:
            with torch.enable_grad():
                x_for_heads = x.detach().requires_grad_(True)
                _, _, latent_tokens = self._compute_score_field(
                    x_for_heads,
                    global_condition,
                    node_mask=node_mask,
                    create_graph=False,
                )
            with torch.no_grad():
                logits_deg = self.degree_head(latent_tokens)
                logits_exist = self.exist_head(latent_tokens).squeeze(-1)

            exist_probs = torch.sigmoid(logits_exist)
            x[..., 0] = (exist_probs >= exist_threshold).float()
            self._last_deg_classes = torch.argmax(logits_deg, dim=-1).detach().cpu()

        return x.detach()


class EqMConditionalNodeGenerator(ConditionalNodeGeneratorBase):
    """Scikit-learn friendly facade for a conditional EqM node generator."""

    def __init__(
        self,
        latent_embedding_dimension: int = 128,
        number_of_transformer_layers: int = 4,
        transformer_attention_head_count: int = 4,
        transformer_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        maximum_epochs: int = 10,
        batch_size: int = 32,
        total_steps: int = 100,
        verbose: bool = False,
        important_feature_index: int = 1,
        lambda_degree_importance: float = 1.0,
        noise_degree_factor: float = 2.0,
        degree_temperature: Optional[float] = None,
        lambda_node_exist_importance: float = 1.0,
        default_exist_pos_weight: float = 1.0,
        lambda_locality_importance: float = 1.0,
        use_guidance: bool = False,
        pool_condition_tokens: bool = False,
        use_locality_supervision: bool = False,
        eqm_sigma: float = 0.2,
        sampling_step_size: float = 0.05,
        sampling_steps: Optional[int] = None,
        langevin_noise_scale: float = 0.0,
    ):
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.learning_rate = learning_rate
        self.maximum_epochs = maximum_epochs
        self.batch_size = batch_size
        self.total_steps = int(total_steps)
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.lambda_degree_importance = lambda_degree_importance
        self.noise_degree_factor = float(noise_degree_factor)
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.default_exist_pos_weight = default_exist_pos_weight
        self.lambda_locality_importance = lambda_locality_importance
        self.use_guidance = bool(use_guidance)
        self.pool_condition_tokens = bool(pool_condition_tokens)
        self.use_locality_supervision = bool(use_locality_supervision)
        self.eqm_sigma = float(eqm_sigma)
        self.sampling_step_size = float(sampling_step_size)
        self.sampling_steps = int(sampling_steps if sampling_steps is not None else total_steps)
        self.langevin_noise_scale = float(langevin_noise_scale)

        self.number_of_rows_per_example = None
        self.input_feature_dimension = None
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.D_max = None
        self.condition_token_count = 1
        self.condition_feature_dimension = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.use_guidance:
            raise ValueError("EqMConditionalNodeGenerator phase 1 does not implement classifier guidance.")

    def _fit_scalers(self, X_array: np.ndarray, y_array: np.ndarray) -> None:
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.x_scaler.fit(X_array.reshape(-1, X_array.shape[2]))
        if y_array.ndim == 3:
            self.y_scaler.fit(y_array.reshape(-1, y_array.shape[-1]))
        else:
            self.y_scaler.fit(y_array)

    def _transform_data(self, X_array: np.ndarray, y_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, row_count, feature_count = X_array.shape
        X_scaled = self.x_scaler.transform(X_array.reshape(-1, feature_count)).reshape(batch_size, row_count, feature_count)
        if y_array.ndim == 3:
            cond_batch, token_count, cond_dim = y_array.shape
            y_scaled = self.y_scaler.transform(y_array.reshape(-1, cond_dim)).reshape(cond_batch, token_count, cond_dim)
        else:
            y_scaled = self.y_scaler.transform(y_array)
        return X_scaled, y_scaled

    def _inverse_transform_input(self, X_array: np.ndarray) -> np.ndarray:
        batch_size, row_count, feature_count = X_array.shape
        X_orig = self.x_scaler.inverse_transform(X_array.reshape(-1, feature_count)).reshape(batch_size, row_count, feature_count)
        X_orig[..., self.important_feature_index] = np.clip(
            X_orig[..., self.important_feature_index],
            0,
            self.D_max,
        )
        return X_orig

    def setup(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None,
    ):
        effective_locality = self.use_locality_supervision and edge_pairs is not None and edge_targets is not None
        if self.use_locality_supervision and not effective_locality and self.verbose:
            print("Locality supervision requested but edge_pairs/edge_targets not provided; continuing without it.")
        if not effective_locality:
            edge_pairs = None
            edge_targets = None

        max_num_rows = max(x.shape[0] for x in node_encodings_list)
        self.number_of_rows_per_example = max_num_rows

        padded_examples = []
        for x in node_encodings_list:
            if x.shape[0] < max_num_rows:
                x = np.pad(x, ((0, max_num_rows - x.shape[0]), (0, 0)), mode="constant", constant_values=0)
            padded_examples.append(x)

        X_array = np.stack(padded_examples, axis=0)
        y_array = np.array(conditional_graph_encodings)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        if y_array.ndim == 2:
            self.condition_token_count = 1
            self.condition_feature_dimension = y_array.shape[1]
        elif y_array.ndim == 3:
            self.condition_token_count = y_array.shape[1]
            self.condition_feature_dimension = y_array.shape[2]
        else:
            raise ValueError(
                "conditional_graph_encodings must be array-like of shape (B, C) or (B, M, C); "
                f"received shape {y_array.shape}"
            )

        self._fit_scalers(X_array, y_array)

        exist_mask = X_array[..., 0] >= 0.5
        ones = int(exist_mask.sum())
        zeros = int(exist_mask.size) - ones
        if ones == 0:
            exist_pos_weight = float(self.default_exist_pos_weight)
        elif zeros > ones:
            exist_pos_weight = float(zeros) / float(ones)
        else:
            exist_pos_weight = 1.0

        deg_column = X_array[..., self.important_feature_index].reshape(-1, 1)
        degree_scaler = MinMaxScaler().fit(deg_column)
        deg_min_val = degree_scaler.data_min_[0]
        deg_range_val = degree_scaler.data_range_[0]
        if deg_range_val == 0:
            deg_range_val = 1e-8

        X_scaled, y_scaled = self._transform_data(X_array, y_array)
        self.input_feature_dimension = X_scaled.shape[2]
        self.D_max = int(X_array[..., self.important_feature_index].max())

        self.model = EqMConditionalNodeGeneratorModule(
            number_of_rows_per_example=self.number_of_rows_per_example,
            input_feature_dimension=self.input_feature_dimension,
            condition_feature_dimension=self.condition_feature_dimension,
            latent_embedding_dimension=self.latent_embedding_dimension,
            number_of_transformer_layers=self.number_of_transformer_layers,
            transformer_attention_head_count=self.transformer_attention_head_count,
            transformer_dropout=self.transformer_dropout,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            important_feature_index=self.important_feature_index,
            max_degree=self.D_max,
            lambda_degree_importance=self.lambda_degree_importance,
            degree_temperature=self.degree_temperature,
            degree_min_val=deg_min_val,
            degree_range_val=deg_range_val,
            lambda_node_exist_importance=self.lambda_node_exist_importance,
            use_locality_supervision=effective_locality,
            lambda_locality_importance=self.lambda_locality_importance,
            exist_pos_weight=exist_pos_weight,
            eqm_sigma=self.eqm_sigma,
            sampling_step_size=self.sampling_step_size,
            sampling_steps=self.sampling_steps,
            langevin_noise_scale=self.langevin_noise_scale,
            pool_condition_tokens=self.pool_condition_tokens,
        )
        self.model.noise_degree_factor = self.noise_degree_factor

    def fit(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None,
    ):
        X_array = np.stack(
            [
                np.pad(x, ((0, self.number_of_rows_per_example - x.shape[0]), (0, 0)), mode="constant", constant_values=0)
                for x in node_encodings_list
            ],
            axis=0,
        )
        y_array = np.array(conditional_graph_encodings)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        X_scaled, y_scaled = self._transform_data(X_array, y_array)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        mask_array = np.zeros((len(node_encodings_list), self.number_of_rows_per_example), dtype=bool)
        for index, x in enumerate(node_encodings_list):
            mask_array[index, :x.shape[0]] = True
        if node_mask is not None:
            mask_array = np.asarray(node_mask, dtype=bool)
        mask_tensor = torch.tensor(mask_array, dtype=torch.bool)

        effective_locality = self.use_locality_supervision and edge_pairs is not None and edge_targets is not None
        if effective_locality:
            dataset = GraphWithEdgesDataset(
                X_scaled,
                y_scaled,
                edge_pairs,
                edge_targets,
                mask_array,
            )
            dataset_size = len(node_encodings_list)
            train_size = int(0.9 * dataset_size)
            val_size = dataset_size - train_size
            indices = torch.randperm(dataset_size).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_graph_with_edges,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_graph_with_edges,
            )
        else:
            dataset = TensorDataset(X_tensor, y_tensor, mask_tensor)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        trainer = pl.Trainer(
            max_epochs=self.maximum_epochs,
            callbacks=[MetricsLogger()],
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        if not self.verbose:
            with suppress_output():
                trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def predict(
        self,
        conditional_graph_encodings: Any,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
    ) -> List[np.ndarray]:
        if desired_class is not None and self.verbose:
            print("EqMConditionalNodeGenerator phase 1 ignores desired_class because classifier guidance is not implemented.")

        self.device = next(self.model.parameters()).device
        cond_array = np.asarray(conditional_graph_encodings)
        if cond_array.ndim == 1:
            cond_array = cond_array[:, None]
        if cond_array.ndim == 3:
            cond_scaled = self.y_scaler.transform(
                cond_array.reshape(-1, cond_array.shape[-1])
            ).reshape(cond_array.shape)
        else:
            cond_scaled = self.y_scaler.transform(cond_array)
        cond_tensor = torch.tensor(cond_scaled, dtype=torch.float32, device=self.device)

        generated = self.model.generate(
            cond_tensor,
            total_steps=self.sampling_steps,
            desired_class=desired_class,
            use_heads_projection=True,
        )

        gen_np = generated.detach().cpu().numpy()
        gen_orig = self._inverse_transform_input(gen_np)

        deg_classes = getattr(self.model, "_last_deg_classes", None)
        if deg_classes is not None:
            deg_classes = deg_classes.cpu().numpy()
            for index in range(len(gen_orig)):
                gen_orig[index][..., self.important_feature_index] = np.clip(deg_classes[index], 0, None)

        return gen_orig

    def plot_metrics(self, window: int = 10, alpha: float = 0.3):
        if self.model is None:
            print("Model is not fitted yet.")
            return
        plot_metrics(
            train_metrics={
                "total": self.model.train_losses,
                "deg_ce": self.model.train_deg_ce,
                "eqm": self.model.train_recon,
                "exist": self.model.train_exist,
                **({"locality": self.model.train_edge_loss} if self.model.use_locality_supervision else {}),
            },
            val_metrics={
                "total": self.model.val_losses,
                "deg_ce": self.model.val_deg_ce,
                "eqm": self.model.val_recon,
                "exist": self.model.val_exist,
                **({"locality": self.model.val_edge_loss} if self.model.use_locality_supervision else {}),
            },
            window=window,
            alpha=alpha,
        )
