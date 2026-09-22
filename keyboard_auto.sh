/workspace/isaaclab/isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent_auto.py \
    --task=LeIsaac-XTrainer-PickCube-v0 \
    --teleop_device=bi_keyboard \
    --auto --arm=left --num_demos=5 \
    --num_envs=1 --device=cuda --enable_cameras --multi_view \
    --record --dataset_file=./datasets/auto_pick_cube_left.hdf5 \
    --step_hz=30

/workspace/isaaclab/isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent_auto.py \
    --task=LeIsaac-XTrainer-PickCube-v0 \
    --teleop_device=bi_keyboard \
    --auto --arm=right --num_demos=5 \
    --num_envs=1 --device=cuda --enable_cameras --multi_view \
    --record --dataset_file=./datasets/auto_pick_cube_right.hdf5 \
    --step_hz=30
