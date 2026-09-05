"""Stationary recurrent extension of the original scalar-energy NodeField."""
import math
import time
from dataclasses import asdict
import torch
from torch import nn
from .recurrent_diagnostics import (
    RecurrentNodeFieldState, RecurrentNodeFieldTrajectory, detached,
    state_diagnostics, prediction_deltas,
)
from .recurrent_interventions import normalize_interventions, apply_recurrent_intervention


def configure_recurrence(owner, latent_dimension, sigma, **options):
    mode = options["node_field_mode"]
    if mode not in {"baseline", "recurrent_energy"}:
        raise ValueError(f"Unsupported node_field_mode: {mode}")
    options["recurrent_hidden_dimension"] = options["recurrent_hidden_dimension"] if options["recurrent_hidden_dimension"] is not None else latent_dimension
    options["recurrent_sigma_max"] = options["recurrent_sigma_max"] if options["recurrent_sigma_max"] is not None else sigma
    for key in ("recurrent_hidden_dimension", "recurrent_training_steps", "recurrent_detach_interval"):
        value = options[key]
        if key == "recurrent_detach_interval" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("recurrent_sigma_min", "recurrent_sigma_max", "recurrent_loss_discount"):
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if mode == "recurrent_energy" and options["recurrent_sigma_min"] > options["recurrent_sigma_max"]:
        raise ValueError("recurrent_sigma_min must not exceed recurrent_sigma_max")
    if not math.isfinite(options["recurrent_update_scale"]):
        raise ValueError("recurrent_update_scale must be finite")
    if options["recurrent_initial_state"] != "zeros":
        raise ValueError("Only zeros recurrent initialization is supported")
    if options["recurrent_corruption_schedule"] not in {"annealed", "constant", "none"}:
        raise ValueError("Unknown recurrent corruption schedule")
    for key, value in options.items():
        setattr(owner, key, value)


class RecurrentNodeFieldMixin:
    def _initialize_recurrent_modules(self, latent_dimension):
        hidden = self.recurrent_hidden_dimension
        self.recurrent_hidden_projection = nn.Linear(hidden, latent_dimension)
        self.recurrent_input_fusion = nn.Sequential(nn.Linear(2*latent_dimension, latent_dimension), nn.GELU(), nn.LayerNorm(latent_dimension))
        self.recurrent_state_head = nn.Sequential(nn.LayerNorm(latent_dimension), nn.Linear(latent_dimension, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.recurrent_state_norm = nn.LayerNorm(hidden) if self.recurrent_state_normalization else nn.Identity()

    def _initialize_recurrent_state(self, batch_size, node_count, device, dtype):
        return torch.zeros(batch_size, node_count, self.recurrent_hidden_dimension, device=device, dtype=dtype)

    def _encode_recurrent_with_condition(self, input_rows, hidden_state, global_condition_vector, node_mask=None):
        if node_mask is not None:
            input_rows = input_rows * node_mask.unsqueeze(-1)
            hidden_state = hidden_state * node_mask.unsqueeze(-1)
        x_latent = self.linear_encoder_input_to_latent(self.layernorm_in(input_rows))
        h_latent = self.recurrent_hidden_projection(hidden_state)
        latent = self.recurrent_input_fusion(torch.cat([x_latent, h_latent], dim=-1))
        return self._run_conditioned_transformer(latent, global_condition_vector, node_mask)

    def _update_recurrent_hidden(self, latent_tokens, hidden_state, node_mask=None):
        h = self.recurrent_state_norm(hidden_state + self.recurrent_update_scale * self.recurrent_state_head(latent_tokens))
        return h if node_mask is None else h * node_mask.unsqueeze(-1)

    def _compute_recurrent_score_field(self, noisy_input, hidden_state, global_condition_vector, node_mask=None, *, create_graph):
        with torch.enable_grad():
            # Differentiate at a new x branch, holding h fixed even when callers share ancestors.
            # The clone preserves outer training gradients without including h in this partial.
            noisy_input = noisy_input.clone().requires_grad_(True)
            z = self._encode_recurrent_with_condition(noisy_input, hidden_state, global_condition_vector, node_mask)
            per_node = self.potential_head(z).squeeze(-1)
            if node_mask is not None:
                per_node = per_node * node_mask
            phi = per_node.sum(1)
            score = -torch.autograd.grad(phi.sum(), noisy_input, create_graph=create_graph, retain_graph=True)[0]
            h_next = self._update_recurrent_hidden(z, hidden_state, node_mask)
        return score, phi, z, h_next

    def _build_recurrent_sigma_schedule(self, steps, device, dtype):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.recurrent_corruption_schedule == "none":
            return torch.zeros(steps, device=device, dtype=dtype)
        if self.recurrent_corruption_schedule == "constant" or steps == 1:
            return torch.full((steps,), self.recurrent_sigma_max, device=device, dtype=dtype)
        return torch.exp(torch.linspace(math.log(self.recurrent_sigma_max), math.log(self.recurrent_sigma_min), steps, device=device, dtype=dtype))

    def _recurrent_node_field_loss(self, input_examples, global_condition, node_presence_mask=None,
                                   node_degree_targets=None, node_label_targets=None, *, create_graph,
                                   apply_sparse_supervision=False, pair_targets=None, count_condition=None):
        h = self._initialize_recurrent_state(*input_examples.shape[:2], input_examples.device, input_examples.dtype)
        schedule = self._build_recurrent_sigma_schedule(self.recurrent_training_steps, input_examples.device, input_examples.dtype)
        totals, weights, step_scores = {}, [], {}
        steps = self.recurrent_training_steps
        # Log-space normalization avoids overflow for long rollouts and discount > 1.
        indices = list(range(steps)) if self.recurrent_supervise_all_steps else [steps-1]
        log_weights = input_examples.new_tensor([(steps-1-k)*math.log(self.recurrent_loss_discount) for k in indices])
        normalized = log_weights.softmax(0)
        for k, sigma in enumerate(schedule):
            eps = torch.randn_like(input_examples)
            noisy = (input_examples + sigma * eps).detach().requires_grad_(True)
            score, _, latent_noisy, h_next = self._compute_recurrent_score_field(noisy, h, global_condition, node_presence_mask, create_graph=create_graph)
            if self.recurrent_corruption_schedule == "none":
                score_loss = input_examples.new_zeros(())
            else:
                mask = self._build_node_field_score_mask(input_examples, node_presence_mask, apply_sparse_supervision=apply_sparse_supervision)
                score_loss = ((score + eps/sigma).square()*mask).sum()/mask.sum().clamp_min(1.)
            if k in {0, steps-1} or (k > 0 and k & (k-1) == 0):
                step_scores[f"recurrent_step_{k}_score"] = score_loss.detach()
            if k in indices:
                denoised = noisy + sigma.square()*score
                clean_latent = self._encode_recurrent_with_condition(denoised, h, global_condition, node_presence_mask)
                losses = self._node_structural_losses(clean_latent, score_loss, input_examples, node_presence_mask, node_degree_targets, node_label_targets)
                latent = clean_latent if self.use_locality_supervision else latent_noisy
                if pair_targets is not None:
                    losses["total"], extra = self._additional_structural_losses(losses["total"], latent,
                        global_condition if count_condition is None else count_condition, node_presence_mask, *pair_targets)
                    losses.update(extra)
                w = normalized[len(weights)]
                weights.append(w)
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0) + w * value
            h = h_next
            if not create_graph or (self.recurrent_detach_interval is not None and (k+1) % self.recurrent_detach_interval == 0):
                h = h.detach()
        totals.update(step_scores)
        totals["recurrent_total"] = totals["total"]
        totals["recurrent_score"] = totals["node_field"]
        return totals, latent

    def _recurrent_batch_step(self, batch, *, training):
        pairwise = self.use_locality_supervision or self.use_edge_label_head or self.use_auxiliary_locality_supervision
        x, condition = batch[:2]
        if pairwise:
            pairs = tuple(batch[2:8])
            mask, degree = batch[8:10]
            labels = batch[10] if self.use_node_label_head else None
        else:
            empty = torch.empty((0, 3), device=x.device, dtype=torch.long)
            values = torch.empty(0, device=x.device)
            pairs = (empty, values, empty, values.long(), empty, values)
            mask, degree = batch[2:4]
            labels = batch[4] if self.use_node_label_head else None
        with torch.enable_grad():
            losses, _ = self._recurrent_node_field_loss(x, self._apply_cfg_dropout(condition) if training else condition,
                mask, degree, labels, create_graph=training, apply_sparse_supervision=training,
                pair_targets=pairs, count_condition=condition)
        prefix = "train_" if training else "val_"
        for key, value in losses.items():
            name = "node_label_ce" if key == "label_ce" else key
            self.log(prefix+name, value, on_step=False, on_epoch=True, batch_size=x.shape[0], prog_bar=key == "total")
        return losses["total"] if training else losses["total"].detach()

    def recurrent_readout(self, x, h, condition, node_mask=None):
        """Read a state without advancing its memory or consuming sampling noise."""
        with torch.no_grad():
            z = self._encode_recurrent_with_condition(x, h, condition, node_mask)
            phi_node = self.potential_head(z).squeeze(-1)
            if node_mask is not None:
                phi_node = phi_node * node_mask
            return dict(node_embeddings=x.detach(), latent_tokens=z,
                exist_logits=self.exist_head(z).squeeze(-1) if self.use_existence_head else None,
                degree_logits=self.degree_head(z),
                node_label_logits=self.node_label_head(z) if self.use_node_label_head else None,
                edge_probabilities=self._compute_edge_probability_matrices(z) if self.use_locality_supervision else None,
                edge_label_logits=self._compute_edge_label_logits(z) if self.use_edge_label_head else None,
                phi=phi_node.sum(1))

    def generate_recurrent(self, global_condition, total_steps=None, desired_target=None, guidance_scale=1.,
                           global_condition_unconditional=None, classifier_guidance_fn=None, classifier_scale=0.,
                           use_heads_projection=False, exist_threshold=.5, *, intervention=None,
                           return_trajectory=False, node_mask=None):
        if getattr(self, "node_field_mode", "baseline") != "recurrent_energy":
            raise ValueError("generate_recurrent requires recurrent_energy mode")
        steps = self.sampling_steps if total_steps is None else total_steps
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("total_steps must be a positive integer")
        if guidance_scale < 0 or classifier_scale < 0:
            raise ValueError("Guidance scales must be nonnegative")
        use_cfg = global_condition_unconditional is not None
        if use_cfg and global_condition_unconditional.shape != global_condition.shape:
            raise ValueError("Unconditional and conditional tensors must have the same shape")
        if use_cfg and classifier_guidance_fn is not None:
            raise ValueError("CFG and classifier guidance must use separate sampling paths")
        items = normalize_interventions(intervention, steps)
        rngs = [torch.Generator().manual_seed(item.seed if item.seed is not None else 0) for item in items]
        self.eval()
        for name in ("edge_probability_matrices", "edge_existence_probabilities", "edge_label_matrices",
                     "edge_label_logits", "edge_label_probabilities", "horizon_probability_matrices", "node_presence_mask",
                     "node_existence_probabilities", "deg_classes", "node_label_classes", "node_label_logits", "node_label_probabilities"):
            setattr(self, "_last_"+name, None)
        x = torch.randn(global_condition.shape[0], self.number_of_rows_per_example, self.input_feature_dimension,
                        device=global_condition.device, dtype=next(self.parameters()).dtype)
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1)
        h = self._initialize_recurrent_state(*x.shape[:2], x.device, x.dtype)
        hu = h.clone() if use_cfg else None
        trajectory = RecurrentNodeFieldTrajectory() if return_trajectory else None
        if trajectory is not None:
            trajectory.x.append(detached(x)); trajectory.h.append(detached(h))
        previous_score = previous_readout = None
        diagnostic_seconds = 0.
        started = time.perf_counter()
        eta = self.sampling_step_size
        for k in range(steps):
            for item, rng in zip(items, rngs):
                if use_cfg:
                    saved_rng = rng.get_state()
                    _, hu = apply_recurrent_intervention(x, hu, item, k, node_mask, generator=rng)
                    rng.set_state(saved_rng)
                x, h = apply_recurrent_intervention(x, h, item, k, node_mask, generator=rng)
                if trajectory is not None and item.active(k):
                    trajectory.interventions.append(dict(evaluation_step=k, **asdict(item)))
            x = x.detach().requires_grad_(True)
            h = h.detach()
            with torch.enable_grad():
                score, phi, _, hn = self._compute_recurrent_score_field(x, h, global_condition, node_mask, create_graph=False)
                if use_cfg:
                    su, _, _, hun = self._compute_recurrent_score_field(x, hu.detach(), global_condition_unconditional, node_mask, create_graph=False)
                    hu = hun.detach()
                    score = su + guidance_scale*(score-su)
                if classifier_guidance_fn is not None:
                    score = score + classifier_scale*classifier_guidance_fn(x)
            xn = (x + eta*score).detach()
            if self.langevin_noise_scale > 0:
                xn = xn + math.sqrt(2*eta)*self.langevin_noise_scale*torch.randn_like(xn)
            if node_mask is not None:
                xn = xn * node_mask.unsqueeze(-1)
            hn = hn.detach()
            if not all(torch.isfinite(t).all() for t in (xn, hn, score, phi)):
                raise FloatingPointError(f"Nonfinite recurrent sampling state at step {k}")
            if trajectory is not None:
                ds = time.perf_counter()
                readout = detached(self.recurrent_readout(xn, hn, global_condition, node_mask))
                metrics = state_diagnostics(x, h, xn, hn, score, phi, previous_score)
                metrics.update(prediction_deltas(readout, previous_readout)); metrics["step"] = k
                trajectory.diagnostics.append(metrics)
                trajectory.evaluated_x.append(detached(x)); trajectory.evaluated_h.append(detached(h))
                trajectory.x.append(detached(xn)); trajectory.h.append(detached(hn))
                trajectory.score.append(detached(score)); trajectory.phi.append(detached(phi))
                # This sampler has no corruption sigma; replacement distributions are in intervention events.
                trajectory.sigma.append(0.)
                trajectory.readouts.append(readout)
                previous_score, previous_readout = score.detach(), readout
                diagnostic_seconds += time.perf_counter()-ds
            x, h = xn, hn
        if use_heads_projection:
            readout = self.recurrent_readout(x, h, global_condition, node_mask)
            self._cache_generation_heads(readout["latent_tokens"], exist_threshold)
        self._last_recurrent_state = RecurrentNodeFieldState(x.detach(), h.detach())
        self._last_recurrent_metadata = dict(field_evaluations=steps*(2 if use_cfg else 1),
            diagnostic_readouts=steps if return_trajectory else 0, final_readouts=int(use_heads_projection),
            diagnostic_seconds=diagnostic_seconds, runtime_seconds=time.perf_counter()-started,
            langevin_noise_scale=self.langevin_noise_scale, sampling_step_size=eta,
            initial_distribution="standard_normal", sigma_semantics="no sampling corruption schedule")
        if trajectory is not None:
            trajectory.metadata.update(self._last_recurrent_metadata)
            return x.detach(), trajectory
        return x.detach()
