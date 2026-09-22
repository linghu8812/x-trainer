/workspace/isaaclab/isaaclab.sh -p scripts/environments/teleoperation/replay.py \
    --task=LeIsaac-XTrainer-PickCube-v0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --replay_mode=action \
    --dataset_file=./datasets/auto_pick_cube_left.hdf5 \
    --select_episodes 0 \
    --task_type bi_keyboard
