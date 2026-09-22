# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a leisaac teleoperation with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

'''
【第一步：手动标定】已完成，无需再跑

【第二步：自动采集 —— 左臂】
python scripts/environments/teleoperation/teleop_se3_agent_auto.py \
    --task=LeIsaac-XTrainer-PickCube-v0 \
    --teleop_device=bi_keyboard \
    --auto --arm=left --num_demos=20 \
    --num_envs=1 --device=cuda --enable_cameras \
    --multi_view \
    --record --dataset_file=./datasets/auto_pick_left.hdf5

【第二步：自动采集 —— 右臂】
python scripts/environments/teleoperation/teleop_se3_agent_auto.py \
    --task=LeIsaac-XTrainer-PickCube-v0 \
    --teleop_device=bi_keyboard \
    --auto --arm=right --num_demos=20 \
    --num_envs=1 --device=cuda --enable_cameras \
    --multi_view \
    --record --dataset_file=./datasets/auto_pick_right.hdf5
'''

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", choices=['keyboard', 'so101leader', 'bi-so101leader', 'xtrainerleader', 'bi_keyboard', 'xtrainer_vr'], help="Device for interacting with environment")
parser.add_argument("--port", type=str, default='/dev/ttyACM0', help="Port for the teleop device:so101leader, default is /dev/ttyACM0")
parser.add_argument("--left_arm_port", type=str, default='/dev/ttyACM0', help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0")
parser.add_argument("--right_arm_port", type=str, default='/dev/ttyACM1', help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--multi_view", action="store_true", help="whether to enable quality render mode.")
parser.add_argument("--left_disabled", action="store_true", help="whether to enable quality render mode.")

# ===================== 自动采集新增参数 =====================
parser.add_argument("--auto", action="store_true",
                    help="Enable scripted auto-collection (only for bi_keyboard).")
parser.add_argument("--arm", type=str, default="left", choices=["left", "right"],
                    help="Which arm to use for auto-collection.")
parser.add_argument("--keyframes_file", type=str, default=None,
                    help="(Optional) Path to JSON file with keyframes; overrides built-in poses.")
# ===========================================================

# recorder_parameter
parser.add_argument("--record", action="store_true", help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--resume", action="store_true", help="whether to resume recording in the existing dataset file")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")

parser.add_argument("--recalibrate", action="store_true", help="recalibrate SO101-Leader or Bi-SO101Leader")
parser.add_argument("--quality", action="store_true", help="whether to enable quality render mode.")

import os
import time
import json
import torch

################################################################################
############################### VR stereo vision ###############################
import cv2
import numpy as np

image_queue = multiprocessing.Queue(maxsize=2)

def run_flask_server(q, cert_path, key_path):
    from flask import Flask, Response
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app)

    @app.route('/stereo_feed')
    def stereo_feed():
        def generate():
            print("Web client connected to stream.")
            while True:
                try:
                    frame_bytes = q.get(timeout=1.0)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    continue
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run(host='0.0.0.0', port=8444, threaded=True, use_reloader=False, ssl_context=(cert_path, key_path))
################################################################################

# =============================================================================
# ============ 关键帧姿态（左/右臂各 8 组，来自手动采集）====================
# =============================================================================
# 关节顺序（16 维）:
#   [0] J1_1  [1] J1_2  [2] J1_3  [3] J1_4  [4] J1_5  [5] J1_6  [6] J1_7  [7] J1_8   (左臂 + 夹爪)
#   [8] J2_1  [9] J2_2  [10] J2_3 [11] J2_4 [12] J2_5 [13] J2_6 [14] J2_7 [15] J2_8  (右臂 + 夹爪)
# =============================================================================

LEFT_POSES = {
    "home":         np.array([-0.3 , -0.6 , -0.66,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                               0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
    "above_cube":   np.array([-0.3 , -0.96, -0.6 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                               0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
    "around_cube":  np.array([-0.3 , -0.96, -0.6 ,  0.  ,  0.  ,  0.  , -0.04,  0.04,
                               0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
    "grip_cube":    np.array([-0.3 , -0.96, -1.08,  0.  ,  0.  ,  0.  , -0.04,  0.04,
                               0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
    "above_plate":  np.array([-0.24, -1.5 , -1.8 ,  0.  ,  0.  ,  0.  , -0.04,  0.04,
                               0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
    "on_plate":     np.array([-0.24, -1.5 , -1.68,  0.  ,  0.  ,  0.  , -0.04,  0.04,
                               0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
    "release":      np.array([-0.24, -1.5 , -1.68,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                               0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
    "finish":       np.array([-0.24, -1.5 , -1.92,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                               0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
}

RIGHT_POSES = {
    "home":         np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                              -0.24,  1.02, -1.02,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
    "above_cube":   np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                              -0.24,  1.2 , -1.02,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
    "around_cube":  np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                              -0.24,  1.2 , -1.02,  0.  ,  0.  ,  0.  , -0.04,  0.04], dtype=np.float64),
    "grip_cube":    np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                              -0.24,  0.9 , -1.02,  0.  ,  0.  ,  0.  , -0.04,  0.04], dtype=np.float64),
    "above_plate":  np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                              -0.42,  0.24, -0.18,  0.  ,  0.  ,  0.  , -0.04,  0.04], dtype=np.float64),
    "on_plate":     np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                              -0.42,  0.72, -0.18,  0.  ,  0.  ,  0.  , -0.04,  0.04], dtype=np.float64),
    "release":      np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                              -0.42,  0.72, -0.18,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
    "finish":       np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                              -0.42,  0.42, -0.6 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float64),
}


def _mk(name, target, steps, on_done=None):
    return {
        "name": name,
        "target": np.asarray(target, dtype=np.float64),
        "steps": int(steps),
        "on_done": on_done,
    }


def make_arm_keyframes(poses):
    """
    根据姿态字典生成 8 个阶段的关键帧序列。
    阶段: home -> above_cube -> around_cube -> grip_cube
          -> above_plate -> on_plate -> release -> finish
    """
    return [
        _mk("home",         poses["home"],         30),
        _mk("above_cube",   poses["above_cube"],   40),
        _mk("around_cube",  poses["around_cube"],  25),
        _mk("grip_cube",    poses["grip_cube"],    30),
        _mk("above_plate",  poses["above_plate"],  60),
        _mk("on_plate",     poses["on_plate"],     25),
        _mk("release",      poses["release"],      15),
        _mk("finish",       poses["finish"],       20, on_done="N"),
    ]


def _load_keyframes_from_file(path: str):
    with open(path, "r") as f:
        raw = json.load(f)
    return [_mk(item.get("name", f"phase{i}"),
                item["target"],
                item.get("steps", 30),
                item.get("on_done", None))
            for i, item in enumerate(raw)]
# =============================================================================


class RateLimiter:
    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main():  # noqa: C901
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher_args = vars(args_cli)

    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.managers import TerminationTermCfg, DatasetExportMode

    import leisaac.tasks  # noqa: F401

    from leisaac.devices import Se3Keyboard, SO101Leader, BiSO101Leader, XTrainerLeader, BiKeyboard, XTrainerVR
    from leisaac.assets.robots.xtrainer import XTRAINER_FOLLOWER_USD_JOINT_LIMITS
    from leisaac.enhance.managers import StreamingRecorderManager, EnhanceDatasetExportMode
    from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

    # ===================== ScriptedBiKeyboard =====================
    class ScriptedBiKeyboard(BiKeyboard):
        """自动采集版 BiKeyboard：直接对 _absolute_pos 做关键帧线性插值。"""

        def __init__(self, env, sensitivity=0.05, keyframes=None,
                     num_demos=0, verbose=True):
            super().__init__(env, sensitivity=sensitivity)
            self.keyframes = list(keyframes) if keyframes else []
            self.num_demos = int(num_demos)
            self.verbose = verbose

            self.episode_count = 0
            self._phase_idx = 0
            self._step_in_phase = 0
            self._start_pos = None
            self._my_callbacks = {}

        def add_callback(self, key, func):
            super().add_callback(key, func)
            self._my_callbacks[key] = func

        def reset(self):
            super().reset()
            self._phase_idx = 0
            self._step_in_phase = 0
            self._start_pos = None

        def advance(self):
            if self.num_demos > 0 and self.episode_count >= self.num_demos:
                return None
            if self._phase_idx >= len(self.keyframes):
                return None

            kf = self.keyframes[self._phase_idx]
            target = np.asarray(kf["target"], dtype=np.float64)
            steps = max(int(kf.get("steps", 30)), 1)
            on_done = kf.get("on_done", None)
            name = kf.get("name", f"phase{self._phase_idx}")

            if self._step_in_phase == 0:
                self._start_pos = self._absolute_pos.copy()
                self.started = True
                if self.verbose:
                    print(f"[ScriptedBiKeyboard] === phase {self._phase_idx}: "
                          f"{name} ({steps} steps) ===")

            t = (self._step_in_phase + 1) / float(steps)
            interp = (1.0 - t) * self._start_pos + t * target

            for i in range(len(interp)):
                lo, hi = XTRAINER_FOLLOWER_USD_JOINT_LIMITS[self._xtrainer_joint_names_list[i]]
                interp[i] = float(np.clip(interp[i], lo, hi))

            self._absolute_pos[:] = interp
            self._step_in_phase += 1

            if self._step_in_phase >= steps:
                self._phase_idx += 1
                self._step_in_phase = 0
                self._start_pos = None

                if on_done is not None:
                    self._fire(on_done)
                    if on_done == "N":
                        self.episode_count += 1
                        if self.verbose:
                            print(f"[ScriptedBiKeyboard] episode {self.episode_count} done")
                        self._phase_idx = 0
                        self._step_in_phase = 0
                        self._start_pos = None
                        return None

            return super().advance()

        def _fire(self, key):
            cb = self._my_callbacks.get(key)
            if cb is None:
                if self.verbose:
                    print(f"[ScriptedBiKeyboard] warning: no callback for '{key}'")
                return
            cb()
    # ================================================================

    def manual_terminate(env, success):
        if hasattr(env, "termination_manager"):
            if success:
                env.termination_manager.set_term_cfg("success", TerminationTermCfg(
                    func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)))
            else:
                env.termination_manager.set_term_cfg("success", TerminationTermCfg(
                    func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)))
            env.termination_manager.compute()
        elif hasattr(env, "_get_dones"):
            env.cfg.return_success_status = success

    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = 'FXAA'
        env_cfg.sim.render.rendering_mode = 'quality'

    if "BiArm" in task_name:
        assert args_cli.teleop_device == "bi-so101leader", "only support bi-so101leader for bi-arm task"
    is_direct_env = "Direct" in task_name
    if is_direct_env:
        assert args_cli.teleop_device in ["so101leader", "bi-so101leader"], "only support so101leader or bi-so101leader for direct task"
    if "XTrainer" in task_name:
        assert args_cli.teleop_device in ["xtrainerleader", "bi_keyboard", "xtrainer_vr"], "only support xtrainerleader, bi_keyboard or xtrainer_vr for xtrainer task"

    if args_cli.auto:
        assert args_cli.teleop_device == "bi_keyboard", \
            "--auto 只能配合 --teleop_device=bi_keyboard 使用"

    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
    else:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    if args_cli.record:
        if args_cli.resume:
            env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
            assert os.path.exists(args_cli.dataset_file), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
        else:
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
            assert not os.path.exists(args_cli.dataset_file), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    else:
        env_cfg.recorders = None

    env = gym.make(task_name, cfg=env_cfg).unwrapped
    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = 'gzip'

    print(f"--------------------------------------------------")
    print(f"Physics dt: {env.physics_dt:.5f} s")
    print(f"Decimation: {env.cfg.decimation}")
    print(f"Control dt: {env.step_dt:.5f} s")
    print(f"Expected FPS: {1.0/env.step_dt:.1f} Hz (must be the same as step_hz)")
    print(f"--------------------------------------------------")

    # create controller
    if args_cli.teleop_device == "keyboard":
        teleop_interface = Se3Keyboard(env, sensitivity=0.25 * args_cli.sensitivity)
        args_cli.multi_view = False
    elif args_cli.teleop_device == "so101leader":
        teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
        args_cli.multi_view = False
    elif args_cli.teleop_device == "bi-so101leader":
        teleop_interface = BiSO101Leader(env, left_port=args_cli.left_arm_port,
                                          right_port=args_cli.right_arm_port,
                                          recalibrate=args_cli.recalibrate)
        args_cli.multi_view = False
    elif args_cli.teleop_device == "xtrainerleader":
        teleop_interface = XTrainerLeader(env, args_cli.left_disabled)
    elif args_cli.teleop_device == "bi_keyboard":
        if args_cli.auto:
            if args_cli.keyframes_file:
                keyframes = _load_keyframes_from_file(args_cli.keyframes_file)
                print(f"[Auto] 从 {args_cli.keyframes_file} 加载 {len(keyframes)} 个关键帧")
            else:
                poses = LEFT_POSES if args_cli.arm == "left" else RIGHT_POSES
                keyframes = make_arm_keyframes(poses)
                print(f"[Auto] 使用 {args_cli.arm.upper()} 臂内置姿态 ({len(keyframes)} 个阶段)")

            num_demos = args_cli.num_demos if args_cli.num_demos > 0 else 10
            teleop_interface = ScriptedBiKeyboard(
                env,
                sensitivity=0.06 * args_cli.sensitivity,
                keyframes=keyframes,
                num_demos=num_demos,
                verbose=True,
            )
            print(f"[Auto] 使用臂: {args_cli.arm}")
            print(f"[Auto] 目标回合数: {num_demos}")
            if not args_cli.record:
                print("[Auto] ⚠️  未开启 --record，数据不会保存！")
        else:
            teleop_interface = BiKeyboard(env, sensitivity=0.06 * args_cli.sensitivity)
    elif args_cli.teleop_device == "xtrainer_vr":
        teleop_interface = XTrainerVR(env, args_cli.left_disabled)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'so101leader', 'bi-so101leader', 'xtrainerleader', 'bi_keyboard', 'xtrainer_vr'."
        )

    # ---- 注册回调 ----
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)
    print(teleop_interface)

    rate_limiter = RateLimiter(args_cli.step_hz)

    if args_cli.teleop_device == "xtrainer_vr":
        package_root = os.path.dirname(leisaac.__file__)
        XLEVR_PATH = os.path.join(package_root, "xtrainer_utils", "XLeVR")
        cert_path = os.path.join(XLEVR_PATH, 'cert.pem')
        key_path = os.path.join(XLEVR_PATH, 'key.pem')
        if not os.path.exists(cert_path):
            print(f"❌ Error: Certificate file not found {cert_path}")
        flask_process = multiprocessing.Process(
            target=run_flask_server,
            args=(image_queue, cert_path, key_path),
            daemon=True
        )
        flask_process.start()
        print(">>> Stereo Visual Streamer started as a SEPARATE PROCESS at http://[IP]:8444/stereo_feed")

    # reset environment
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop_interface.reset()

    if args_cli.teleop_device == "xtrainer_vr":
        stereo_left_sensor = env.unwrapped.scene["stereo_left"]
        stereo_right_sensor = env.unwrapped.scene["stereo_right"]

    if args_cli.multi_view:
        try:
            import omni.kit.viewport.utility
            import omni.ui
            robot_base_path = "/World/envs/env_0/Robot/x_trainer_asm_0226_SLDASM"
            cameras_config = {
                "Top_View": f"{robot_base_path}/base_link/top_camera",
                "Left_Wrist_View": f"{robot_base_path}/J1_6/left_wrist_camera",
                "Right_Wrist_View": f"{robot_base_path}/J2_6/right_wrist_camera"
            }
            for win_name, cam_path in cameras_config.items():
                vp_win = omni.kit.viewport.utility.create_viewport_window(win_name)
                if vp_win:
                    vp_win.viewport_api.camera_path = cam_path
        except Exception as e:
            print(f"[Warning] Setup error: {e}")

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    start_record_state = False
    ui_frame_count = 0
    layout_done = False

    while simulation_app.is_running():
        ui_frame_count += 1

        if args_cli.multi_view and not layout_done and ui_frame_count == 60:
            try:
                import omni.ui
                top_win = omni.ui.Workspace.get_window("Top_View")
                left_win = omni.ui.Workspace.get_window("Left_Wrist_View")
                right_win = omni.ui.Workspace.get_window("Right_Wrist_View")
                main_dock_space = omni.ui.Workspace.get_window("Viewport")
                if top_win and left_win and right_win and main_dock_space:
                    print("Executing Custom Layout Docking...")
                    top_win.dock_in(main_dock_space, omni.ui.DockPosition.SAME)
                    left_win.dock_in(top_win, omni.ui.DockPosition.BOTTOM, ratio=0.4)
                    right_win.dock_in(left_win, omni.ui.DockPosition.RIGHT, ratio=0.5)
                    top_win.visible = True
                    left_win.visible = True
                    right_win.visible = True
                    print(">>> Windows docked successfully!")
                    layout_done = True
                else:
                    print(f"Waiting for windows... (Main: {main_dock_space is not None})")
            except Exception as e:
                print(f"Layout failed: {e}")
                layout_done = True

        with torch.inference_mode():
            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)

            actions = teleop_interface.advance()

            if should_reset_task_success:
                print("Task Success!!!")
                should_reset_task_success = False
                if args_cli.record:
                    manual_terminate(env, True)

            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False
                if start_record_state:
                    if args_cli.record:
                        print("Stop Recording!!!")
                    start_record_state = False
                if args_cli.record:
                    manual_terminate(env, False)

                if args_cli.record and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

                if args_cli.record and args_cli.num_demos > 0 and \
                   env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break

                if args_cli.teleop_device == "xtrainer_vr":
                    env.sim.render()

            elif actions is None:
                env.render()
            else:
                if not start_record_state:
                    if args_cli.record:
                        print("Start Recording!!!")
                    start_record_state = True
                env.step(actions)

            if args_cli.teleop_device == "xtrainer_vr":
                try:
                    img_l_raw = stereo_left_sensor.data.output["rgb"][0]
                    img_r_raw = stereo_right_sensor.data.output["rgb"][0]
                    sbs_tensor = torch.cat([img_l_raw, img_r_raw], dim=1)
                    sbs_img = sbs_tensor.cpu().numpy().astype(np.uint8)
                    sbs_img_bgr = cv2.cvtColor(sbs_img, cv2.COLOR_RGB2BGR)
                    ret, buffer = cv2.imencode('.jpg', sbs_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        if not image_queue.full():
                            image_queue.put_nowait(frame_bytes)
                        else:
                            try:
                                image_queue.get_nowait()
                                image_queue.put_nowait(frame_bytes)
                            except Exception:
                                pass
                except Exception as e:
                    print(f"Vision error: {e}")

            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
