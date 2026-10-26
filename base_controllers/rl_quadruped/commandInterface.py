"""Operator input for the RL quadruped controller: keyboard, Xbox pad over ``/joy``, or none.

Every backend provides a body-frame velocity command ``(vx, vy, wz)`` and a queue of one-shot
:class:`Event` values.  The emergency ``STOP`` is the space bar on the keyboard (also while typing a
velocity) and BACK or LB+RB on the pad.
"""

import atexit
import os
import select
import subprocess
import sys
import termios
import time
import tty
from enum import Enum

import numpy as np
import rospy as ros
from sensor_msgs.msg import Joy
from termcolor import colored


class Event(Enum):
    CALIBRATE = "calibrate"
    STAND_UP = "stand_up"
    START_RL = "start_rl"
    SAFE_STOP = "safe_stop"    # hand over to the zero-command policy, the robot stays standing
    STAND_DOWN = "stand_down"
    STOP = "stop"              # emergency: collapse on joint damping
    QUIT = "quit"
    PUSH = "push"              # simulation only: one random shove
    PUSH_BURST = "push_burst"  # simulation only: a run of random shoves


class CommandInterfaceBase(object):
    """Velocity command, push magnitude and the event queue."""

    def __init__(self, max_lin_vel=0.5, max_ang_vel=0.5, push_force=125.0, push_force_step=25.0):
        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)
        self.velocity_cmd = np.zeros(3)
        self.push_force = float(push_force)
        self.push_force_step = float(push_force_step)
        self._events = []

    def post(self, event):
        self._events.append(event)

    def poll(self):
        """Return the events seen since the last call, oldest first."""
        events, self._events = self._events, []
        return events

    def update(self):
        pass

    def get_velocity_cmd(self):
        return self.velocity_cmd

    def zero(self):
        self.velocity_cmd[:] = 0.0

    def adjust_push_force(self, delta):
        self.push_force = max(0.0, self.push_force + delta)
        print(colored(f"push force {self.push_force:.0f} N", "cyan"))

    def shutdown(self):
        pass


class KeyboardCommandInterface(CommandInterfaceBase):
    """Non-blocking raw-stdin keyboard.  The command is held between key presses: direction keys
    step it, number keys jump to a forward preset, ``v`` reads ``vx[,vy[,wz]]`` up to Enter."""

    ENTRY_TIMEOUT = 8.0   # s without a keystroke before a half-typed velocity is dropped

    KEY_MAP = {
        "p": Event.PUSH,
        "c": Event.CALIBRATE,
        "u": Event.STAND_UP,
        "r": Event.START_RL,
        "f": Event.SAFE_STOP,
        "d": Event.STAND_DOWN,
        " ": Event.STOP,
        "q": Event.QUIT,
    }

    def __init__(self, max_lin_vel=0.5, max_ang_vel=0.5, lin_step=0.1, ang_step=0.1,
                 speed_presets=(0.0, 0.1, 0.2, 0.3, 0.4), push_force=125.0, push_force_step=25.0):
        super(KeyboardCommandInterface, self).__init__(max_lin_vel, max_ang_vel, push_force,
                                                       push_force_step)
        self.lin_step = lin_step
        self.ang_step = ang_step
        self.speed_presets = tuple(speed_presets)
        self._entry = None          # None when not typing, else the text so far
        self._entry_stamp = 0.0
        self._fd = sys.stdin.fileno()
        if not os.isatty(self._fd):
            raise RuntimeError("Keyboard input needs a terminal; use --input joy or none")
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._restored = False
        atexit.register(self.shutdown)
        print(colored(self.help(), "cyan"))

    def shutdown(self):
        if not self._restored:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._restored = True

    def update(self):
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            if not key:
                break
            self._handle(key)
        if self._entry is not None and time.monotonic() - self._entry_stamp > self.ENTRY_TIMEOUT:
            self._entry = None
            print(colored("\rVelocity entry timed out", "yellow"), flush=True)

    def _handle_entry(self, key):
        if key == " ":
            self._entry = None
            self.post(Event.STOP)
            return
        self._entry_stamp = time.monotonic()
        if key in ("\r", "\n"):
            text, self._entry = self._entry, None
            self._apply_entry(text)
        elif key == "\x1b":                      # Escape
            self._entry = None
            print(colored("\rVelocity entry cancelled", "yellow"), flush=True)
        elif key in ("\x7f", "\b"):              # Backspace
            self._entry = self._entry[:-1]
            print(f"\rvelocity> {self._entry} \b", end="", flush=True)
        elif key in "0123456789+-.,eE":
            self._entry += key
            print(f"\rvelocity> {self._entry}", end="", flush=True)

    def _apply_entry(self, text):
        if not text.strip():
            print(colored("\rVelocity entry cancelled", "yellow"), flush=True)
            return
        try:
            values = [float(part) for part in text.split(",") if part.strip()]
        except ValueError:
            print(colored(f"\rCannot read '{text}' as vx[,vy[,wz]]", "red"), flush=True)
            return
        if not 1 <= len(values) <= 3:
            print(colored(f"\rExpected 1 to 3 values, got {len(values)}", "red"), flush=True)
            return
        self.velocity_cmd[:] = 0.0
        self.velocity_cmd[:len(values)] = values
        requested = self.velocity_cmd.copy()
        self._clamp()
        self._report("typed", requested)

    def _clamp(self):
        np.clip(self.velocity_cmd[:2], -self.max_lin_vel, self.max_lin_vel,
                out=self.velocity_cmd[:2])
        self.velocity_cmd[2] = float(np.clip(self.velocity_cmd[2],
                                             -self.max_ang_vel, self.max_ang_vel))

    def _report(self, source, requested=None):
        cmd = self.velocity_cmd
        text = (f"velocity command  vx={cmd[0]:+.2f} m/s  vy={cmd[1]:+.2f} m/s  "
                f"wz={cmd[2]:+.2f} rad/s   [{source}]")
        if requested is not None and not np.allclose(requested, cmd):
            print(colored(f"\r{text}  <- clamped from {np.round(requested, 2)}", "yellow"),
                  flush=True)
        else:
            print(colored("\r" + text, "green"), flush=True)

    def _handle(self, key):
        if self._entry is not None:
            self._handle_entry(key)
            return
        if key == "v":
            self._entry = ""
            self._entry_stamp = time.monotonic()
            print(colored("\rType vx[,vy[,wz]] then Enter (Esc cancels): ", "cyan"), end="",
                  flush=True)
            return
        if key.isdigit() and key != "0" and int(key) <= len(self.speed_presets):
            self.velocity_cmd[:] = (self.speed_presets[int(key) - 1], 0.0, 0.0)
            self._clamp()
            self._report(f"preset {key}")
            return
        if key == "P":                   # checked before lower-casing
            self.post(Event.PUSH_BURST)
            return
        if key in ("+", "="):
            self.adjust_push_force(self.push_force_step)
            return
        if key == "-":
            self.adjust_push_force(-self.push_force_step)
            return

        lowered = key.lower()
        if lowered == "x":
            self.zero()
            self._report("zeroed")
            return
        if lowered in self.KEY_MAP:
            self.post(self.KEY_MAP[lowered])
            return
        steps = {"w": (0, self.lin_step), "s": (0, -self.lin_step),
                 "a": (1, self.lin_step), "z": (1, -self.lin_step),
                 "j": (2, self.ang_step), "l": (2, -self.ang_step)}
        if lowered in steps:
            axis, delta = steps[lowered]
            self.velocity_cmd[axis] += delta
            requested = self.velocity_cmd.copy()
            self._clamp()
            self._report(f"{lowered} {delta:+.2f}", requested)

    def help(self):
        presets = "  ".join(f"{i + 1}:{v:g}" for i, v in enumerate(self.speed_presets))
        return ("\nKeyboard commands\n"
                f"  c  calibrate IMU bias        w/s  forward / backward (step {self.lin_step:g} m/s)\n"
                "  u  stand up                  a/z  left / right\n"
                f"  r  start RL policy           j/l  turn left / right (step {self.ang_step:g} rad/s)\n"
                "  f  SAFE STOP (stay standing) v    type an exact velocity: vx[,vy[,wz]] Enter\n"
                "  d  stand down                x    zero the velocity command\n"
                f"  SPACE  EMERGENCY damping     forward presets  {presets}\n"
                "  q  quit\n"
                "\nDisturbances (simulation only)\n"
                f"  p  one random push           -/+  push force (now {self.push_force:.0f} N)\n"
                "  P  a burst of random pushes\n")


class JoyCommandInterface(CommandInterfaceBase):
    """Xbox pad over the ROS ``/joy`` topic (xpad layout), with edge-detected buttons."""

    A, B, X, Y, LB, RB, BACK, START = 0, 1, 2, 3, 4, 5, 6, 7

    BUTTON_MAP = {
        Y: Event.CALIBRATE,
        A: Event.STAND_UP,
        X: Event.START_RL,
        B: Event.STAND_DOWN,
        BACK: Event.STOP,
        START: Event.QUIT,
    }

    def __init__(self, max_lin_vel=0.5, max_ang_vel=0.5, dead_zone=0.08, push_force=125.0,
                 push_force_step=25.0):
        super(JoyCommandInterface, self).__init__(max_lin_vel, max_ang_vel, push_force,
                                                  push_force_step)
        self.dead_zone = dead_zone
        self._msg = Joy()
        self._prev_buttons = None
        self._start_joy_node()
        self._sub = ros.Subscriber("/joy", Joy, self._callback, queue_size=1, tcp_nodelay=True)
        print(colored(self.help(), "cyan"))

    @staticmethod
    def _start_joy_node():
        subprocess.run(["rosnode", "kill", "/joy_node"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        subprocess.Popen(["rosrun", "joy", "joy_node", "_dev:=/dev/input/js0"])
        try:
            ros.wait_for_message("/joy", Joy, timeout=5.0)
            print(colored("Joystick connected", "green"))
        except ros.ROSException:
            print(colored("No /joy message within 5 s: check that the pad is plugged in and on",
                          "yellow"))

    def _callback(self, msg):
        self._msg = msg

    def _dead_zone(self, value):
        return value if abs(value) >= self.dead_zone else 0.0

    def update(self):
        msg = self._msg
        if len(msg.axes) >= 4:
            # left stick: translation, right stick x: yaw rate
            self.velocity_cmd[0] = self.max_lin_vel * self._dead_zone(msg.axes[1])
            self.velocity_cmd[1] = self.max_lin_vel * self._dead_zone(msg.axes[0])
            self.velocity_cmd[2] = self.max_ang_vel * self._dead_zone(msg.axes[3])

        buttons = list(msg.buttons)
        if not buttons:
            return
        if self._prev_buttons is None:
            self._prev_buttons = buttons
            return
        pressed = [i for i, b in enumerate(buttons)
                   if b and i < len(self._prev_buttons) and not self._prev_buttons[i]]
        both_bumpers = len(buttons) > self.RB and buttons[self.LB] and buttons[self.RB]
        if both_bumpers:
            if self.LB in pressed or self.RB in pressed:
                self.post(Event.STOP)
        elif self.RB in pressed:
            # A panic grip can close RB one message before LB: the STOP that follows still wins.
            self.post(Event.SAFE_STOP)
        for index in pressed:
            if index in self.BUTTON_MAP:
                self.post(self.BUTTON_MAP[index])
        self._prev_buttons = buttons

    @staticmethod
    def help():
        return ("\nJoystick commands\n"
                "  Y  calibrate IMU bias        left stick   forward / lateral\n"
                "  A  stand up                  right stick X  turn\n"
                "  X  start RL policy\n"
                "  B  stand down                RB  SAFE STOP (stay standing)\n"
                "  BACK or LB+RB  EMERGENCY damping      START  quit\n")


def create_command_interface(kind, cfg):
    """``'keyboard'``, ``'joy'`` or ``'none'`` (scripted runs post their own events)."""
    options = dict(max_lin_vel=cfg["max_lin_vel_cmd"], max_ang_vel=cfg["max_ang_vel_cmd"],
                   push_force=cfg["push_force"], push_force_step=cfg["push_force_step"])
    if kind == "keyboard":
        return KeyboardCommandInterface(lin_step=cfg["key_lin_step"], ang_step=cfg["key_ang_step"],
                                        speed_presets=cfg["key_speed_presets"], **options)
    if kind == "joy":
        return JoyCommandInterface(dead_zone=cfg["joy_dead_zone"], **options)
    if kind == "none":
        return CommandInterfaceBase(**options)
    raise ValueError(f"Unknown command interface '{kind}'")
