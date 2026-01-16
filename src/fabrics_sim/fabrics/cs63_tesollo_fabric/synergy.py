# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch

def build_dg3f_synergy_matrix(num_fingers, joints_per_finger, device, latent_dim=8):
    """
    Builds a synergy projection matrix for the DG3F (Tesollo) gripper.
    
    Joint order: [F1M1,F1M2,F1M3,F1M4, F2M1,F2M2,F2M3,F2M4, F3M1,F3M2,F3M3,F3M4]
    
    Mechanical Structure (per finger):
    - M1: Axial Rotation (Self-rotation) -> Independent per finger
    - M2: Lateral Motion (Side-to-side) -> Independent per finger
    - M3 & M4: Main Envelope joints (Curling/Flexion) -> Global Sync/Shape
    
    Latent Semantics (8-dim):
    - z0, z1, z2: M1 joints for Finger 1, 2, 3 respectively (Independent Axial Rotation).
    - z3, z4, z5: M2 joints for Finger 1, 2, 3 respectively (Independent Lateral Motion).
    - z6: Global Envelope Sync (M3 + M4 across all fingers). Primary wrap/grasp.
    - z7: Global Envelope Shape (M3 - M4 across all fingers). Adjusts tip curvature.
    
    Args:
        num_fingers: 3
        joints_per_finger: 4
        device: Torch device
        latent_dim: 8
    """
    if latent_dim != 8:
        raise ValueError(f"Tesollo independent M1/M2 synergy requires latent_dim=8, got {latent_dim}.")

    num_gripper_joints = num_fingers * joints_per_finger
    P = torch.zeros(latent_dim, num_gripper_joints, device=device)

    # z0, z1, z2: Independent M1 (Axial Rotation)
    for fi in range(num_fingers):
        P[fi, fi * joints_per_finger + 0] = 1.0

    # z3, z4, z5: Independent M2 (Lateral Motion)
    for fi in range(num_fingers):
        P[3 + fi, fi * joints_per_finger + 1] = 1.0

    # z6: Global Envelope Sync (M3 + M4)
    for fi in range(num_fingers):
        base = fi * joints_per_finger
        P[6, base + 2] = 1.0
        P[6, base + 3] = 1.0

    # z7: Global Envelope Shape (M3 - M4)
    for fi in range(num_fingers):
        base = fi * joints_per_finger
        P[7, base + 2] = 1.0
        P[7, base + 3] = -1.0

    # Normalize rows to keep scales comparable.
    P = P / (torch.linalg.norm(P, dim=1, keepdim=True).clamp_min(1e-12))
    return P

def build_dg3f_geometric_taskmap(num_fingers, joints_per_finger, device):
    """
    [TEMPLATE] Proposed geometric task map for the DG3F gripper.
    """
    return None
