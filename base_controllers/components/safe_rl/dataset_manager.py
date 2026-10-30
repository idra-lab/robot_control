import numpy as np
import os
from termcolor import colored

# Per robot.  Fall rule: run to the ground, the walking policy recovered from at most 29 deg of
# inclination (Aliengo, 320 pushed episodes; Go2 15 deg, 200), the base never below 0.27 m (Go2
# 0.26 m), and every fall went past 64 deg (Go2 85 deg): the thresholds sit in those gaps.  The kick
# range spans the robot's fall boundary, with about 45 % of the episodes falling.
ROBOT_SETTINGS = {
    "aliengo": {"INCLINATION_THRESHOLD": 45.0, "FALL_HEIGHT_THRESHOLD": 0.2, "kick_range": (0.0, 3.0)},
    "go2": {"INCLINATION_THRESHOLD": 45.0, "FALL_HEIGHT_THRESHOLD": 0.2, "kick_range": (0.0, 2.5)},
}


# the episodes CSV written next to a dataset, one row per episode (README.md)
EPISODE_COLUMNS = ("episode", "cmd_vx", "cmd_vy", "cmd_wz", "push_time", "kick", "kick_dir", "base_vx", "base_vy",
                   "fallen", "captured", "survived", "end_time", "n_samples", "row", "discarded")


class DatasetManager():
    def __init__(self, quadruped=None):

        self.quadruped = quadruped
        # -------------------------------
        # Simulation Thresholds and Constants
        # -------------------------------
        # per robot, ROBOT_SETTINGS
        self.INCLINATION_THRESHOLD = 45.0  # degrees - max allowed inclination before considering robot as fallen
        self.FALL_HEIGHT_THRESHOLD = 0.2  # meters - min allowed height before considering robot as fallen
        self.CP_SAFE_RADIUS = 0.05  # meters - acceptable radius to consider CP successful
        # s the capture point has to stay in that radius: a robot can stand still for 0.2 s on its
        # feet and then fall (2 of 91 captures at 0.2 s, none of 54 at 0.5 s)
        self.CAPTURE_HOLD = 0.5
        self.G = 9.81  # gravitational constant
        self.policy_frequency = 50  # Hz
        self.decimation = (1 / self.quadruped.dt) * (1 / self.policy_frequency)

        # -------------------------------
        # Episode: reset, walk at a command, push, recover or fall
        # -------------------------------
        # Velocity command (vx, vy, wz), drawn as the walking policy's training draws it (safe_rl
        # velocity_env_cfg.py, CommandsCfg): uniform in +-cmd_range, rel_standing of the episodes at
        # zero and rel_yaw_only of the others only turning.  Never outside the training range.
        self.cmd_range = np.array([0.5, 0.5, 0.5])
        self.rel_standing = 0.1
        self.rel_yaw_only = 0.1
        # The push starts at a time uniform in push_window, s after the policy starts: the gait has
        # settled by then (~1 s) and the window randomises its phase.  It is the push benchmark's
        # (rl_quadruped/push_sweep.py), so kicks compare with its k50: a horizontal force of
        # robot mass x kick / push_duration on the base for push_duration s, the kick uniform in
        # kick_range m/s, the direction uniform in the base frame.
        self.push_window = (2.0, 3.0)
        self.kick_range = (0.0, 3.0)  # per robot, ROBOT_SETTINGS
        self.push_duration = 0.2
        # The episode ends at a fall, at capture (capture point within CP_SAFE_RADIUS for
        # CAPTURE_HOLD s, from the end of the push) or with the push survived for post_push_time s
        # after its end.
        self.post_push_time = 3.0
        # None: the walking policy throughout.  Else the safe policy takes over, at zero command,
        # backup_delay s after the push starts.
        self.backup_delay = None
        if quadruped.robot_name not in ROBOT_SETTINGS:
            raise ValueError(f"no dataset settings for {quadruped.robot_name}: add it to ROBOT_SETTINGS")
        for name, value in ROBOT_SETTINGS[quadruped.robot_name].items():
            setattr(self, name, value)
        self.episode = {}

    # -------------------------------
    # Capture Point Computation
    # -------------------------------
    def compute_capture_point(self, pos, vel, height):
        tc = np.sqrt(height / self.G)  # time constant based on height
        return pos + vel * tc

    # -------------------------------
    # Fall Condition Checker
    # -------------------------------
    def check_fallen(self, base_z_position, inclination_deg):
        return inclination_deg > self.INCLINATION_THRESHOLD or base_z_position < self.FALL_HEIGHT_THRESHOLD

    def check_termination(self):
        # -------------------------------
        # Check Falling
        # -------------------------------
        inclination_after = 2 * np.arcsin(
            np.sqrt(self.quadruped.quaternion[0] ** 2 + self.quadruped.quaternion[1] ** 2)) * (180 / np.pi)
        if self.check_fallen(self.quadruped.basePoseW[2], inclination_after):
            self.fallen_flag = 1

        # -------------------------------
        # Check com inside safe radius of capture point
        # -------------------------------
        base_pos_xy = self.quadruped.basePoseW[:2].copy()
        base_vel_xy = self.quadruped.baseTwistW[:2].copy()
        z_after = self.quadruped.basePoseW[2].copy()
        cp = self.compute_capture_point(base_pos_xy, base_vel_xy, z_after)
        cp_local = cp - base_pos_xy

        # when self.capture_flag = 1 it means we are stable: the capture point within CP_SAFE_RADIUS
        # for CAPTURE_HOLD s.  A single tick inside is not enough, the base can be momentarily still
        # while the robot tips over its feet.
        if np.linalg.norm(
                cp_local) < self.CP_SAFE_RADIUS and self.quadruped.time >= self.capture_from and self.fallen_flag == 0:
            if self.cp_inside_since is None:
                self.cp_inside_since = self.quadruped.time
            if self.quadruped.time - self.cp_inside_since >= self.CAPTURE_HOLD - 1e-6:
                self.capture_flag = 1
        else:
            self.cp_inside_since = None
        # survived the push
        if self.quadruped.time >= self.end_time - 1e-6 and self.fallen_flag == 0 and self.capture_flag == 0:
            self.survived_flag = 1

        self.quadruped.ros_pub.add_marker(np.append(base_pos_xy, 0.), radius=0.1, color="blue")
        self.quadruped.ros_pub.add_marker(np.append(cp, 0.), radius=0.1, color="red")
        return self.capture_flag or self.survived_flag or self.fallen_flag

    def store_observations(self):
        # -------------------------------
        # Save Observation Transition for VF Learning
        # -------------------------------
        if self.step % self.decimation == 0:
            body_ang_vel = self.quadruped.b_R_w @ self.quadruped.baseTwistW[3:]
            proj_gravity = self.quadruped.b_R_w.dot(np.array([0, 0, -1]))
            # dimension of observation vector = 3+3+12+12=30
            self.obs_tp1_ = np.concatenate((
                proj_gravity.astype(np.float32),
                body_ang_vel.astype(np.float32),
                self.quadruped.q.astype(np.float32),
                self.quadruped.qd.astype(np.float32)
            ))
            # dimension of full_obs vector = 30+30+1+1=62; the last is success: captured, or the
            # push survived for post_push_time
            success = float(self.capture_flag or self.survived_flag)
            full_obs = np.concatenate([self.obs_t_, self.obs_tp1_, [float(self.fallen_flag), success]])
            # collect data from the end of the push on: V of a state is whether the robot recovers
            # from it with no further push.  Recorded from the push start, a walking state about to
            # be pushed carried the push's outcome, and walking itself was learnt as unsafe.
            if self.quadruped.time >= self.record_from:
                self.observations.append(full_obs)

            # store last observation
            self.obs_t_ = self.obs_tp1_

    def wait_for_reset(self, base_pose, q_reset, timeout=3.0, still_time=0.1):
        """Hold q_reset until the measurements show the reset robot standing still.  The reset goes
        through Gazebo services while the subscribers still hold the previous episode's state, and
        the teleport spikes the accelerometer and leaves the joints moving: started at once, the
        episode flags the previous fall at t = 0 and seeds the policy with garbage.  The base back
        at its reset position tells the state is fresh; then it has to be still for still_time."""
        robot = self.quadruped
        fresh, still_ticks = False, 0
        for _ in range(int(timeout / robot.dt)):
            if not fresh:
                fresh = (np.linalg.norm(robot.basePoseW[:3] - base_pose[:3]) < 0.02
                         and np.all(np.isfinite(robot.baseLinAccB)))
            else:
                still = (np.max(np.abs(robot.qd)) < 0.3 and np.linalg.norm(robot.baseTwistW) < 0.1
                         and abs(np.linalg.norm(robot.baseLinAccB) - self.G) < 0.5)
                still_ticks = still_ticks + 1 if still else 0
                if still_ticks * robot.dt >= still_time:
                    return True
            robot.send_des_jstate(q_reset, np.zeros(12), np.zeros(12))
            robot.rate.sleep()
        print(colored("The reset robot did not settle; starting anyway", "red"))
        return False

    def joint_range(self):
        """Position limits of the joints, in the order of quadruped.q."""
        model = self.quadruped.robot.model
        idx = [model.idx_qs[model.getJointId(name)] for name in self.quadruped.joint_names]
        return model.lowerPositionLimit[idx], model.upperPositionLimit[idx]

    def sample_command(self):
        """A velocity command (vx, vy, wz) drawn as in the walking policy's training."""
        cmd = np.random.uniform(-self.cmd_range, self.cmd_range)
        if np.random.uniform() < self.rel_standing:
            cmd[:] = 0.0
        elif np.random.uniform() < self.rel_yaw_only:
            cmd[:2] = 0.0
        return cmd

    # -------------------------------
    # Main Function: Single Simulation Episode
    # -------------------------------
    def run_single_simulation(self, actor_network, noise_std=1.0):
        robot = self.quadruped
        # init vars
        self.fallen_flag = 0
        self.capture_flag = 0
        self.survived_flag = 0
        self.cp_inside_since = None
        self.observations = []
        self.obs_t_ = []
        self.obs_tp1_ = []

        actor_network.velocity_cmd = self.sample_command()
        # the push, on the control tick grid
        push_step = int(round(np.random.uniform(*self.push_window) / robot.dt))
        push_time = round(push_step * robot.dt, 4)
        kick = np.random.uniform(*self.kick_range)
        kick_dir = np.random.uniform(-np.pi, np.pi)  # base frame, 0 forward, pi/2 left
        # the torque noise starts with the push, the samples once it is over
        self.warmup_time = push_time
        self.capture_from = push_time + self.push_duration
        self.record_from = self.capture_from
        self.end_time = self.capture_from + self.post_push_time
        backup_time = np.inf if self.backup_delay is None else push_time + self.backup_delay
        print(f"Nominal policy velocity command {actor_network.velocity_cmd}, push at {push_time} s, "
              f"kick {kick:.2f} m/s at {np.degrees(kick_dir):.0f} deg")
        # what the episode was, saved with the dataset for dataset_statistics.py
        self.episode = {"cmd": actor_network.velocity_cmd.copy(), "push_time": push_time, "kick": kick,
                        "kick_dir": kick_dir, "base_vel": np.full(2, np.nan), "fell_before_push": False}

        max_steps = int(round(self.end_time / robot.dt)) + 2 * int(self.decimation)
        for self.step in range(max_steps):
            robot.updateKinematics()
            terminate = self.check_termination()
            if self.fallen_flag and self.step < push_step:
                print(colored(f"Fallen at {robot.time} s, before the push", "red"))
                self.episode["fell_before_push"] = True
                break
            self.store_observations()
            # Stop at a fall, capture or survival, on a sample so that it carries the label
            if robot.time >= self.warmup_time and terminate and self.step % self.decimation == 0:
                print(f"Termination at {robot.time}")
                break

            if self.step == push_step:
                yaw = robot.basePoseW[5]
                force = robot.robot.robotMass * kick / self.push_duration
                robot.push(force * np.array([np.cos(yaw + kick_dir), np.sin(yaw + kick_dir)]), self.push_duration)
                self.episode["base_vel"] = (robot.b_R_w @ robot.baseTwistW[:3])[:2]
                print(f"Pushing robot at {push_time} s: {force:.0f} N for {self.push_duration} s")

            # -------------------------------
            # Observation and Action Computation
            # -------------------------------
            lin_acc_b = robot.baseLinAccB
            lin_vel_b = robot.b_R_w.dot(robot.baseTwistW[:3])
            ang_vel_b = robot.b_R_w.dot(robot.baseTwistW[3:6])
            proj_gravity = robot.b_R_w.dot(np.array([0, 0, -1]))

            if robot.time < backup_time:  # nominal policy
                if np.mod(robot.time, 0.5) == 0:
                    print(colored(f"TIME: {robot.time}", "blue"))
                robot.q_des = actor_network.action(lin_acc_b, lin_vel_b, ang_vel_b, proj_gravity,
                                                   robot.q, robot.qd, policy_type="default")
            else:  # switch to backup policy
                if np.mod(robot.time, 0.5) == 0:
                    print(colored(f"TIME: {robot.time}", "red"))
                actor_network.velocity_cmd = np.array([0.0, 0.0, 0.0])
                robot.q_des = actor_network.action(lin_acc_b, lin_vel_b, ang_vel_b, proj_gravity,
                                                   robot.q, robot.qd, policy_type="safe")

            # for log
            robot.baseTwistW_des[:3] = robot.b_R_w.T @ np.append(actor_network.velocity_cmd[:2], 0.0)
            robot.baseTwistW_des[5] = actor_network.velocity_cmd[2]

            # Add noise to simulate real-world actuation
            robot.tau_ffwd = np.zeros(12)
            if robot.time >= self.warmup_time:
                robot.tau_ffwd += np.random.normal(0, noise_std, size=robot.tau_ffwd.shape)
            # switch off wbc
            robot.grForcesW_des = np.zeros((12))

            # send to PD controller
            robot.send_des_jstate(robot.q_des, np.zeros(12), robot.tau_ffwd)
            robot.rate.sleep()
            robot.time = np.round(robot.time + robot.dt, 4)
            robot.ros_pub.publishVisual(delete_markers=True)

        return np.array(self.observations), self.fallen_flag, self.capture_flag, self.survived_flag

    # -------------------------------
    # Run a Batch of Simulations and Save Results
    # -------------------------------
    def run_batch_simulations(self, actor_network, n_episodes=100, save_path="results", noise_std=1.0, seed=None):
        os.makedirs(save_path, exist_ok=True)
        all_obs = []
        stats = []
        episodes = []  # one row per episode, saved next to the dataset
        max_len = 0

        if seed is not None:
            np.random.seed(seed)

        print("Running batch simulations...")
        # joint readings must stay in the robot's range, on the branch its encoders report: a reading
        # 2 pi off (a wrapped joint) would put states in the dataset the real robot never shows.
        # 0.5 rad of slack covers the soft joint stops overshooting in a hard fall.
        q_lower, q_upper = self.joint_range()
        # every episode starts standing still in the policy's default posture, feet 3 mm above the
        # ground (standingHeight is to the foot centres)
        q_reset = np.array(actor_network.q_def, dtype=np.float64)
        base_pose = np.array([0, 0, self.quadruped.standingHeight(q_reset) + 0.03, 0., 0., 0.])

        for i in (range(n_episodes)):
            print(colored(f"Simulation {i}-----------------------", "blue"))
            # -------------------------------
            # Reset base
            # -------------------------------
            self.quadruped.q_des = q_reset.copy()
            # the reset reads the joints the previous episode left, so it comes before initVars; it
            # leaves the stand-alone joint gains on: on the policy's the robot sags to the fall height
            self.quadruped.resetRobot(basePoseDes=base_pose, q_reset=q_reset)
            self.quadruped.initVars()  # resets time etc
            self.wait_for_reset(base_pose, q_reset)
            # RlVelocityController sets no gains itself
            self.quadruped.pid.setPDjoints(actor_network.kp, actor_network.kd, np.zeros(12))
            # history, previous action and decimation phase start over, seeded from the reset robot
            actor_network.reset()

            obs, fallen, captured, survived = self.run_single_simulation(actor_network=actor_network,
                                                                         noise_std=noise_std)
            print(colored(f"Fallen {fallen}, Captured {captured}, Survived {survived}", "green"))
            ep = self.episode
            row = dict(episode=i, push_time=ep["push_time"], kick=ep["kick"], kick_dir=ep["kick_dir"], fallen=fallen,
                       captured=captured, survived=survived, end_time=float(np.ravel(self.quadruped.time)[0]),
                       n_samples=len(obs), row=len(all_obs), discarded=0)
            row.update(zip(("cmd_vx", "cmd_vy", "cmd_wz"), ep["cmd"]), base_vx=ep["base_vel"][0], base_vy=ep["base_vel"][1])
            episodes.append(row)
            if ep["fell_before_push"] or not len(obs):
                row.update(row=-1, discarded=2)
                continue
            q = np.concatenate([obs[:, 6:18], obs[:, 36:48]])
            if np.any(q < q_lower - 0.5) or np.any(q > q_upper + 0.5):
                print(colored(f"Simulation {i}: joint positions outside the joint range, episode "
                              f"discarded", "red"))
                row.update(row=-1, discarded=1)
                continue
            # the episode is one row, from the push end to the episode end, the fall labelled on the last sample:
            # through the TD target every state of a falling episode learns that it leads there
            all_obs.append(obs)
            max_len = max(max_len, len(obs))
            stats.append((fallen, captured, survived, len(obs)))

        if not all_obs:
            print(colored("No episode kept, nothing saved", "red"))
            return
        # Pad observation arrays to same length in case of early termination (com comverget to cop)
        obs_dim = all_obs[0].shape[1]
        padded_obs = np.zeros((len(all_obs), max_len, obs_dim), dtype=np.float32)

        for i, episode in enumerate(all_obs):
            padded_obs[i, :len(episode), :] = episode

        stats = np.array(stats, dtype=int)

        np.save(os.path.join(save_path, "observations_dataset_abs_test.npy"), padded_obs)
        # row indexes the dataset's first axis (-1: discarded); kick_dir and base velocity at the push
        # in the base frame; discarded 1: joint range, 2: fell before the push or no sample recorded
        np.savetxt(os.path.join(save_path, "observations_dataset_abs_test_episodes.csv"),
                   np.array([[row[c] for c in EPISODE_COLUMNS] for row in episodes], dtype=float),
                   delimiter=",", fmt="%.6g", comments="", header=",".join(EPISODE_COLUMNS))

        print(f"Episodi completati: {n_episodes}")
        print(f"Caduti: {np.sum(stats[:, 0])}, CP raggiunto: {np.sum(stats[:, 1])}, "
              f"sopravvissuti: {np.sum(stats[:, 2])}")
        print(f"Dati salvati in: {save_path}")
        print(f"Shape of observations: {padded_obs.shape}")
