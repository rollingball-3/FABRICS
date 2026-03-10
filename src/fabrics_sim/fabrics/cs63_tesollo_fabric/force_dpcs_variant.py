from __future__ import annotations

import torch

from .fingertip_attractor_variant import CS63TesolloFingertipAttractorFabric


class CS63TesolloForceDPCSFabric(CS63TesolloFingertipAttractorFabric):
    """Scheme-B minimal compatible variant.

    Instead of introducing a new palm-center task map, this variant reuses the
    existing 7-point palm pose task and only shifts the palm translation target
    by `delta = C * F_ext`.
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
        self._scheme_b_enabled = False
        self._scheme_b_force_world = torch.zeros(batch_size, 3, device=device)
        self._scheme_b_compliance = torch.zeros(batch_size, 3, 3, device=device)
        self._scheme_b_shift = torch.zeros(batch_size, 3, device=device)
        self._scheme_b_virtual_center = torch.zeros(batch_size, 3, device=device)
        self._scheme_b_m_min = float(self.fabric_params["palm_attractor"]["min_isotropic_mass"])
        self._scheme_b_m_max = float(self.fabric_params["palm_attractor"]["max_isotropic_mass"])
        self._scheme_b_metric_alpha = float(self.fabric_params["palm_attractor"]["mass_sharpness"])
        self._scheme_b_potential_gain = float(self.fabric_params["palm_attractor"]["conical_gain"])
        self._scheme_b_potential_alpha = float(self.fabric_params["palm_attractor"]["conical_sharpness"])
        self._base_palm_params = {
            "min_isotropic_mass": float(self.fabric_params["palm_attractor"]["min_isotropic_mass"]),
            "max_isotropic_mass": float(self.fabric_params["palm_attractor"]["max_isotropic_mass"]),
            "mass_sharpness": float(self.fabric_params["palm_attractor"]["mass_sharpness"]),
            "mass_switch_offset": float(self.fabric_params["palm_attractor"]["mass_switch_offset"]),
            "conical_sharpness": float(self.fabric_params["palm_attractor"]["conical_sharpness"]),
            "conical_gain": float(self.fabric_params["palm_attractor"]["conical_gain"]),
        }

    def set_scheme_b_config(
        self,
        enabled: bool,
        compliance_matrix: torch.Tensor | None = None,
        m_min: float | None = None,
        m_max: float | None = None,
        metric_alpha: float | None = None,
        potential_gain: float | None = None,
        potential_alpha: float | None = None,
    ) -> None:
        self._scheme_b_enabled = bool(enabled)
        if compliance_matrix is not None:
            compliance = compliance_matrix.to(self.device, dtype=torch.float32)
            if compliance.dim() == 2:
                compliance = compliance.unsqueeze(0).repeat(self.batch_size, 1, 1)
            self._scheme_b_compliance = compliance
        if m_min is not None:
            self._scheme_b_m_min = float(m_min)
        if m_max is not None:
            self._scheme_b_m_max = float(m_max)
        if metric_alpha is not None:
            self._scheme_b_metric_alpha = float(metric_alpha)
        if potential_gain is not None:
            self._scheme_b_potential_gain = float(potential_gain)
        if potential_alpha is not None:
            self._scheme_b_potential_alpha = float(potential_alpha)
        # For the minimal-compatible Scheme-B path we must preserve the nominal
        # zero-force behavior. Rewriting the 7-point palm attractor parameters
        # changes the closed-loop system even when F_ext = 0, so here we only
        # cache Scheme-B parameters and use them for diagnostics/tuning, while
        # the actual control action remains the virtual-center translation shift.

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
        scheme_b_enable: bool | None = None,
        scheme_b_force_world: torch.Tensor | None = None,
        scheme_b_compliance: torch.Tensor | None = None,
        scheme_b_m_min: float | None = None,
        scheme_b_m_max: float | None = None,
        scheme_b_metric_alpha: float | None = None,
        scheme_b_potential_gain: float | None = None,
        scheme_b_potential_alpha: float | None = None,
    ):
        if (
            scheme_b_enable is not None
            or scheme_b_compliance is not None
            or scheme_b_m_min is not None
            or scheme_b_m_max is not None
            or scheme_b_metric_alpha is not None
            or scheme_b_potential_gain is not None
            or scheme_b_potential_alpha is not None
        ):
            self.set_scheme_b_config(
                enabled=self._scheme_b_enabled if scheme_b_enable is None else bool(scheme_b_enable),
                compliance_matrix=self._scheme_b_compliance if scheme_b_compliance is None else scheme_b_compliance,
                m_min=self._scheme_b_m_min if scheme_b_m_min is None else float(scheme_b_m_min),
                m_max=self._scheme_b_m_max if scheme_b_m_max is None else float(scheme_b_m_max),
                metric_alpha=(
                    self._scheme_b_metric_alpha if scheme_b_metric_alpha is None else float(scheme_b_metric_alpha)
                ),
                potential_gain=(
                    self._scheme_b_potential_gain if scheme_b_potential_gain is None else float(scheme_b_potential_gain)
                ),
                potential_alpha=(
                    self._scheme_b_potential_alpha
                    if scheme_b_potential_alpha is None
                    else float(scheme_b_potential_alpha)
                ),
            )

        shifted_palm_position = palm_position
        if scheme_b_force_world is not None:
            force_world = scheme_b_force_world.to(self.device, dtype=torch.float32)
            if force_world.dim() == 1:
                force_world = force_world.unsqueeze(0).repeat(self.batch_size, 1)
            self._scheme_b_force_world = force_world
        else:
            self._scheme_b_force_world.zero_()

        if self._scheme_b_enabled:
            self._scheme_b_shift = torch.bmm(
                self._scheme_b_compliance,
                self._scheme_b_force_world.unsqueeze(2),
            ).squeeze(2)
            shifted_palm_position = palm_position + self._scheme_b_shift
            self._scheme_b_virtual_center = shifted_palm_position
        else:
            self._scheme_b_shift.zero_()
            self._scheme_b_virtual_center = palm_position

        super().set_features(
            fingertip_targets=fingertip_targets,
            palm_position=shifted_palm_position,
            palm_matrix=palm_matrix,
            batched_cspace_position=batched_cspace_position,
            batched_cspace_velocity=batched_cspace_velocity,
            object_ids=object_ids,
            object_indicator=object_indicator,
            cspace_damping_gain=cspace_damping_gain,
            force_scale=force_scale,
        )

    def get_last_scheme_b_shift(self) -> torch.Tensor:
        return self._scheme_b_shift

    def get_last_scheme_b_virtual_center(self) -> torch.Tensor:
        return self._scheme_b_virtual_center

    def get_last_scheme_b_force_world(self) -> torch.Tensor:
        return self._scheme_b_force_world

