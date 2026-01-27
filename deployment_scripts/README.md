# CS63 Tesollo Fabrics Control (deployment_scripts)

这些脚本用于 **CS63 机械臂 + Tesollo(DG3F) 手爪** 在实物/ROS2 上联调与部署 Fabrics 控制。

## 脚本清单与应用场景

- **`cs63_tesollo_fabric_node.py`**
  - **场景**：真机 Fabrics 控制闭环的主节点（核心）。
  - **订阅**（默认）：
    - `/cs63/joint_states`：机械臂关节反馈（JointState）
    - `/tesollo/joint_states`：手爪关节反馈（JointState）
    - `/cs63_tesollo_fabric/pose_commands`：palm 目标（JointState.position = `[x,y,z,roll,pitch,yaw]`，弧度）
    - `/cs63_tesollo_fabric/tracked_points_commands`：点目标（flatten N*3，相对 `palm_link`）
  - **发布**（默认）：
    - `/cs63/joint_commands`：机械臂关节命令（position/velocity）
    - `/tesollo/joint_commands`：手爪关节命令（position/velocity）
    - `/cs63_tesollo_fabric/joint_states`：Fabrics 状态输出（position/velocity/effort = q/qd/qdd）
  - **用途**：评估 Fabrics 在实物上的稳定性、响应、振荡、限幅、安全等。

- **`cs63_tesollo_random_targets.py`**
  - **场景**：最简单的联调工具，持续发送 palm pose 与 tracked points 的随机/周期目标。
  - **用途**：验证“命令能进 fabric node、fabric node 能出 joint_commands、真机能动”。

- **`cs63_tesollo_state_machine.py`**
  - **场景**：极简状态机（集成测试），周期性发目标；可选从 `/tf` 读取 `obj_pos` 做简单 y 方向跟随。
  - **用途**：验证 TF/感知接线与话题链路，不依赖学习策略。

- **`cs63_tesollo_camera_transform_publisher.py`**
  - **场景**：发布相机外参 TF（robot_base -> camera frame）。
  - **用途**：你做视觉闭环/策略输入需要统一 TF 树时使用。
  - **备注**：支持 `--matrix-file` 读取 4x4 `robot_T_camera`。

- **`cs63_tesollo_image_subscriber.py`**
  - **场景**：订阅并可视化深度图（检查相机数据是否正常）。
  - **用途**：相机联调、确认编码/分辨率/对齐等。

## 真机底层需要具备的最小能力

- **反馈**：arm 与 gripper 都能持续发布 `sensor_msgs/JointState`（至少 position）到对应 topic。
- **控制**：arm 与 gripper 都能接收 `JointState` 形式的 position（建议也接收 velocity）并执行 **position control**。
- **安全**：驱动侧具备限幅/软限位/急停等兜底（强烈建议）。
- **关节一致性**：关节 name/顺序与脚本一致；否则用 `--arm-joint-names` / `--gripper-joint-names` 显式指定。

