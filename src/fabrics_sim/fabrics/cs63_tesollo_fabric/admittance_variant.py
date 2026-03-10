# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import torch

from .fingertip_attractor_variant import CS63TesolloFingertipAttractorFabric


class CS63TesolloAdmittanceFabric(CS63TesolloFingertipAttractorFabric):
    """Fingertip-attractor fabric with Forced Admittance term.

    Scheme A:
        qdd = M^{-1}[-(nominal_terms) + tau_adm]
    In natural form used by this codebase:
        qdd = -M^{-1} f
    Therefore, we inject admittance by shifting force:
        f <- f - tau_adm
    """

    def __init__(
        self,
        batch_size,
        device,
        timestep,
        default_joint_config,
        num_arm_joints=6,
        num_gripper_joints=12,
        num_fingers=3,
        graph_capturable=True,
    ):
        super().__init__(
            batch_size=batch_size,
            device=device,
            timestep=timestep,
            default_joint_config=default_joint_config,
            num_arm_joints=num_arm_joints,
            num_gripper_joints=num_gripper_joints,
            num_fingers=num_fingers,
            graph_capturable=graph_capturable,
        )
        self._adm_enabled = False
        self._adm_tau_clip: float | None = None
        self._adm_filter_alpha: float = 0.0
        self._tau_ext_joint = torch.zeros(batch_size, self.num_joints, device=device)
        self._tau_adm = torch.zeros_like(self._tau_ext_joint)

    def set_admittance_config(
        self,
        enabled: bool,
        tau_clip: float | None = None,
        filter_alpha: float = 0.0,
    ) -> None:
        self._adm_enabled = bool(enabled)
        self._adm_tau_clip = None if tau_clip is None else float(tau_clip)
        self._adm_filter_alpha = float(max(0.0, min(1.0, filter_alpha)))

    def _compute_tau_adm(
        self,
        tau_ext_joint: torch.Tensor | None,
        K_adm: torch.Tensor | float | None,
    ) -> torch.Tensor:
        if tau_ext_joint is None:
            tau = torch.zeros_like(self._tau_ext_joint)
        else:
            tau = tau_ext_joint.to(self._tau_ext_joint.device, dtype=self._tau_ext_joint.dtype)
            if tau.dim() == 1:
                tau = tau.unsqueeze(0).repeat(self.batch_size, 1)
        if not self._adm_enabled:
            return torch.zeros_like(self._tau_adm)

        if K_adm is None:
            tau_adm_raw = tau
        elif isinstance(K_adm, (float, int)):
            tau_adm_raw = float(K_adm) * tau
        else:
            K = K_adm.to(self._tau_ext_joint.device, dtype=self._tau_ext_joint.dtype)
            if K.dim() == 0:
                tau_adm_raw = K.item() * tau
            elif K.dim() == 1:
                tau_adm_raw = tau * K.unsqueeze(0)
            elif K.dim() == 2:
                if K.shape[0] == self.num_joints and K.shape[1] == self.num_joints:
                    tau_adm_raw = torch.bmm(K.unsqueeze(0).repeat(self.batch_size, 1, 1), tau.unsqueeze(2)).squeeze(2)
                else:
                    tau_adm_raw = tau * K
            elif K.dim() == 3:
                tau_adm_raw = torch.bmm(K, tau.unsqueeze(2)).squeeze(2)
            else:
                raise ValueError(f"Unsupported K_adm shape: {tuple(K.shape)}")

        if self._adm_tau_clip is not None and self._adm_tau_clip > 0.0:
            tau_adm_raw = torch.clamp(tau_adm_raw, -self._adm_tau_clip, self._adm_tau_clip)

        if self._adm_filter_alpha > 0.0:
            self._tau_adm = self._adm_filter_alpha * self._tau_adm + (1.0 - self._adm_filter_alpha) * tau_adm_raw
        else:
            self._tau_adm = tau_adm_raw
        return self._tau_adm

    def set_features(
        self,
        fingertip_targets,
        palm_position,
        palm_matrix,
        batched_cspace_position,
        batched_cspace_velocity,
        object_ids,
        object_indicator,
        cspace_damping_gain=None,
        force_scale=None,
        tau_ext_joint: torch.Tensor | None = None,
        K_adm: torch.Tensor | float | None = None,
        adm_enable: bool | None = None,
        adm_tau_clip: float | None = None,
        adm_filter_alpha: float | None = None,
    ):
        if adm_enable is not None or adm_tau_clip is not None or adm_filter_alpha is not None:
            self.set_admittance_config(
                enabled=self._adm_enabled if adm_enable is None else bool(adm_enable),
                tau_clip=self._adm_tau_clip if adm_tau_clip is None else float(adm_tau_clip),
                filter_alpha=self._adm_filter_alpha if adm_filter_alpha is None else float(adm_filter_alpha),
            )

        super().set_features(
            fingertip_targets=fingertip_targets,
            palm_position=palm_position,
            palm_matrix=palm_matrix,
            batched_cspace_position=batched_cspace_position,
            batched_cspace_velocity=batched_cspace_velocity,
            object_ids=object_ids,
            object_indicator=object_indicator,
            cspace_damping_gain=cspace_damping_gain,
            force_scale=force_scale,
        )
        self._tau_ext_joint = tau_ext_joint if tau_ext_joint is not None else torch.zeros_like(self._tau_ext_joint)
        self._compute_tau_adm(self._tau_ext_joint, K_adm)

    def forward(self, q, qd, timestep):
        metric, force, metric_inv = super().forward(q, qd, timestep)
        if self._adm_enabled:
            force = force - self._tau_adm
        return metric, force, metric_inv

    def get_last_tau_adm(self) -> torch.Tensor:
        return self._tau_adm

