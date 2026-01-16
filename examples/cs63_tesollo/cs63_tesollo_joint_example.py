# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time
import argparse
import torch
import numpy as np

from fabrics_sim.fabrics.cs63_tesollo_fabric import CS63TesolloGripperJointAttractorFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

"""
Example usage:
python cs63_tesollo_joint_example.py --batch_size=2 --cuda_graph
"""

def main():
    parser = argparse.ArgumentParser(description='CS63-Tesollo Joint fabric example.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
    parser.add_argument('--cuda_graph', action='store_true', help='Enable CUDA graph.')
    args = parser.parse_args()

    device = 'cuda:0'
    initialize_warp("0")

    world_model = WorldMeshesModel(batch_size=args.batch_size, max_objects_per_env=20, 
                                   device=device, world_filename='floor')
    object_ids, object_indicator = world_model.get_object_ids()

    control_rate = 60.
    timestep = 1./control_rate
    
    q_start = torch.zeros(18, device=device)
    q_start[1] = -1.57
    
    # 12D Joint variant
    fabric = CS63TesolloGripperJointAttractorFabric(args.batch_size, device, timestep, 
                                                    default_joint_config=q_start)
    integrator = DisplacementIntegrator(fabric)

    q = q_start.unsqueeze(0).repeat(args.batch_size, 1).contiguous()
    qd = torch.zeros_like(q)
    qdd = torch.zeros_like(q)

    # Joint targets: (b, 12)
    joint_targets = torch.zeros(args.batch_size, 12, device=device)
    palm_pos = torch.tensor([-0.4, 0.0, 0.5], device=device).repeat(args.batch_size, 1)
    palm_rot = torch.eye(3, device=device).unsqueeze(0).repeat(args.batch_size, 1, 1)
    damping_gain = 10.0 * torch.ones(args.batch_size, 1, device=device)

    if args.cuda_graph:
        # Note: joint_targets is the first feature
        inputs = [joint_targets, palm_pos, palm_rot, q.detach(), qd.detach(), 
                  object_ids, object_indicator, damping_gain]
        g, q_new, qd_new, qdd_new = capture_fabric(fabric, q, qd, qdd, timestep, integrator, inputs, device)

    start = time.time()
    for i in range(200):
        if i % 50 == 0:
            joint_targets.uniform_(-0.5, 0.5)
            palm_pos += torch.randn_like(palm_pos) * 0.05

        if args.cuda_graph:
            g.replay()
            q.copy_(q_new); qd.copy_(qd_new); qdd.copy_(qdd_new)
        else:
            fabric.set_features(joint_targets, palm_pos, palm_rot, q.detach(), qd.detach(), 
                                object_ids, object_indicator, damping_gain)
            q, qd, qdd = integrator.step(q.detach(), qd.detach(), qdd.detach(), timestep)

        if i % 20 == 0:
            print(f"Step {i}, q[0, 6:18] (gripper): {q[0, 6:18].cpu().numpy()}")

    print(f"Done in {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
