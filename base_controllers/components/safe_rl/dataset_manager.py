import numpy as np
import os
from termcolor import colored


class DatasetManager():
    def __init__(self, quadruped=None):

        self.quadruped = quadruped
        # -------------------------------
        # Simulation Thresholds and Constants
        # -------------------------------
        self.INCLINATION_THRESHOLD = 30.0  # degrees - max allowed inclination before considering robot as fallen
        self.FALL_HEIGHT_THRESHOLD = 0.2  # meters - min allowed height before considering robot as fallen
        self.CP_SAFE_RADIUS = 0.05  # meters - acceptable radius to consider CP successful
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
        self.kick_range = (0.0, 3.0)
        self.push_duration = 0.2

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

        # when self.capture_flag = 1 it means we are stable
        if np.linalg.norm(
                cp_local) < self.CP_SAFE_RADIUS and self.quadruped.time >= self.warmup_time and self.fallen_flag == 0:
            self.capture_flag = 1

        self.quadruped.ros_pub.add_marker(np.append(base_pos_xy, 0.), radius=0.1, color="blue")
        self.quadruped.ros_pub.add_marker(np.append(cp, 0.), radius=0.1, color="red")
        return self.capture_flag or self.fallen_flag

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
            # dimension of full_obs vector = 30+30+1+1=62
            full_obs = np.concatenate([self.obs_t_, self.obs_tp1_, [float(self.fallen_flag), float(self.capture_flag)]])
            # print('store',self.capture_flag)
            # collect data only for backup policy which is active after warmup (the = ensures at least one sample is collected before break)
            if self.quadruped.time >= self.warmup_time:
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
    def run_single_simulation(self, actor_network, max_steps=8000, noise_std=1.0):
        # init vars
        self.fallen_flag = 0
        self.capture_flag = 0
        self.first_time = True
        self.observations = []
        self.obs_t_ = []
        self.obs_tp1_ = []

        actor_network.velocity_cmd = self.sample_command()
        # the push, on the control tick grid
        push_step = int(round(np.random.uniform(*self.push_window) / self.quadruped.dt))
        push_time = round(push_step * self.quadruped.dt, 4)
        kick = np.random.uniform(*self.kick_range)
        kick_dir = np.random.uniform(-np.pi, np.pi)  # base frame, 0 forward, pi/2 left
        # samples are recorded from the push on
        self.warmup_time = push_time
        print(f"Nominal policy velocity command {actor_network.velocity_cmd}, push at {push_time} s, "
              f"kick {kick:.2f} m/s at {np.degrees(kick_dir):.0f} deg")

        for self.step in range(max_steps):
            self.quadruped.updateKinematics()
            terminate = self.check_termination()
            self.store_observations()
            if self.quadruped.time > self.warmup_time:
                # Stop if CP was reached successfully
                if terminate and self.step % self.decimation == 0:
                    # print('terminate', self.capture_flag)
                    print(f"Termination at {self.quadruped.time}")
                    break

            if self.step == push_step:
                yaw = self.quadruped.basePoseW[5]
                force = self.quadruped.robot.robotMass * kick / self.push_duration
                self.quadruped.push(force * np.array([np.cos(yaw + kick_dir), np.sin(yaw + kick_dir)]),
                                    self.push_duration)
                print(f"Pushing robot at {push_time} s: {force:.0f} N for {self.push_duration} s")
            # -------------------------------
            # Observation and Action Computation
            # -------------------------------
            # Prepare observation
            lin_acc_b = self.quadruped.baseLinAccB
            lin_vel_b = self.quadruped.b_R_w.dot(self.quadruped.baseTwistW[:3])
            ang_vel_b = self.quadruped.b_R_w.dot(self.quadruped.baseTwistW[3:6])
            proj_gravity = self.quadruped.b_R_w.dot(np.array([0, 0, -1]))

            # Inference (get new action)
            # Generate random initial command (velocity and yaw) to explore nominal policy states
            if True:#self.quadruped.time <= warmup_time:  # nominal policy
                if np.mod(self.quadruped.time, 0.5) == 0:
                    print(colored(f"TIME: {self.quadruped.time}", "blue"))

                self.quadruped.q_des = actor_network.action(lin_acc_b, lin_vel_b, ang_vel_b, proj_gravity,
                                                            self.quadruped.q, self.quadruped.qd, policy_type="default")

            else:  # switch to backup policy
                if np.mod(self.quadruped.time, 0.5) == 0:
                    print(colored(f"TIME: {self.quadruped.time}", "red"))
                actor_network.velocity_cmd = np.array([0.0, 0.0, 0.0])
                self.quadruped.q_des = actor_network.action(lin_acc_b, lin_vel_b, ang_vel_b, proj_gravity,
                                                            self.quadruped.q, self.quadruped.qd, policy_type="safe")

            # for log
            self.quadruped.baseTwistW_des[:3] = self.quadruped.b_R_w.T @ np.append(actor_network.velocity_cmd[:2], 0.0)
            self.quadruped.baseTwistW_des[5] = actor_network.velocity_cmd[2]

            # Add noise to simulate real-world actuation
            self.quadruped.tau_ffwd = np.zeros(12)

            if self.quadruped.time >= self.warmup_time:
                noise = np.random.normal(0, noise_std, size=self.quadruped.tau_ffwd.shape)
                self.quadruped.tau_ffwd += noise
            # switch off wbc
            self.quadruped.grForcesW_des = np.zeros((12))

            # send to PD controller
            self.quadruped.send_des_jstate(self.quadruped.q_des, np.zeros(12), self.quadruped.tau_ffwd)
            self.quadruped.rate.sleep()
            self.quadruped.time = np.round(self.quadruped.time + self.quadruped.dt, 4)  # np.array([self.loop_time]), 3)
            self.quadruped.ros_pub.publishVisual(delete_markers=True)

        return np.array(self.observations), self.fallen_flag, self.capture_flag

    # -------------------------------
    # Run a Batch of Simulations and Save Results
    # -------------------------------
    def run_batch_simulations(self, actor_network, n_episodes=100, save_path="results", noise_std=1.0, seed=None):
        os.makedirs(save_path, exist_ok=True)
        all_obs = []
        stats = []
        max_len = 0

        if seed is not None:
            np.random.seed(seed)

        print("Running batch simulations...")
        # every episode starts standing still in the policy's default posture, feet 3 mm above the
        # ground (standingHeight is to the foot centres)
        q_reset = np.array(actor_network.q_def, dtype=np.float64)
        base_pose = np.array([0, 0, self.quadruped.standingHeight(q_reset) + 0.03, 0., 0., 0.])

        n_obs_data = n_episodes
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

            obs, fallen, captured = self.run_single_simulation(actor_network=actor_network, noise_std=noise_std)
            print(colored(f"Fallen {fallen}, Captured {captured}", "green"))
            #print('obs', obs)
            if fallen and len(obs) > 10:
                n_obs_data += 1
                obs_fall = obs[-10:]
                obs_no_fall = obs[:-10]

                #print('obs_fall', obs_fall)
                #print('obs_no_fall', obs_no_fall)
                all_obs.append(obs_fall)
                all_obs.append(obs_no_fall)

                max_len = max(max_len, len(obs_fall))
                max_len = max(max_len, len(obs_no_fall))

            else:
                all_obs.append(obs)

                # debug
                # print(obs.shape)

                max_len = max(max_len, len(obs))

            stats.append((fallen, captured, len(obs)))

        # Pad observation arrays to same length in case of early termination (com comverget to cop)
        obs_dim = all_obs[0].shape[1]
        padded_obs = np.zeros((n_obs_data, max_len, obs_dim), dtype=np.float32)

        for i, episode in enumerate(all_obs):
            padded_obs[i, :len(episode), :] = episode

        stats = np.array(stats, dtype=int)

        np.save(os.path.join(save_path, "observations_dataset_abs_test.npy"), padded_obs)

        print(f"Episodi completati: {n_episodes}")
        print(f"Caduti: {np.sum(stats[:, 0])}, CP raggiunto: {np.sum(stats[:, 1])}")
        print(f"Dati salvati in: {save_path}")
        print(f"Shape of observations: {padded_obs.shape}")
