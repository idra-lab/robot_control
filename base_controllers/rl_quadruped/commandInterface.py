"""Operator input for the RL quadruped controller: keyboard or Xbox joystick, one API.

Both backends expose the same two things:

* a body-frame velocity command ``(vx, vy, wz)``, already scaled and dead-zoned;
* a queue of one-shot :class:`Event` values, so the caller sees a button press exactly once.

The keyboard backend reads raw stdin, so it needs no window, no X server and no extra packages -
handy over ssh to the robot.  The joystick backend rides on the ``/joy`` topic published by the ROS
``joy`` node, which is what the lab's Xbox pads are already wired to.

The ``STOP`` event is the emergency one; on the pad it is deliberately mapped to a button you can hit
without aiming (both bumpers, or Back), and on the keyboard to the space bar.  The space bar keeps
that meaning at all times, including while a velocity is being typed - which is why the numeric entry
mode uses commas as separators and never spaces.

The keyboard offers three ways to set a velocity: stepping it with the direction keys, jumping to a
preset with the number keys, and typing an exact command after ``v``.
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
    """One-shot operator requests."""

    CALIBRATE = "calibrate"
    STAND_UP = "stand_up"
    START_RL = "start_rl"
    SAFE_STOP = "safe_stop"  # hand over to the zero-command policy; the robot stays standing
    STAND_DOWN = "stand_down"
    STOP = "stop"          # emergency: collapse on joint damping
    QUIT = "quit"
    ZERO_CMD = "zero_cmd"
    PUSH = "push"              # simulation only: shove the base once, random direction
    PUSH_BURST = "push_burst"  # simulation only: a run of random shoves


class CommandInterfaceBase(object):
    """Shared velocity-command state and event queue."""

    def __init__(self, max_lin_vel=0.5, max_ang_vel=0.5, push_force=125.0,
                 push_force_step=25.0):
        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)
        self.velocity_cmd = np.zeros(3)
        # Magnitude of the disturbance the PUSH events ask for, in newtons.  It lives here beside
        # the velocity command because it is an operator setting, not a robot property: the
        # controller reads it when a push is requested and turns it into a wrench.
        self.push_force = float(push_force)
        self.push_force_step = float(push_force_step)
        self._events = []

    def _post(self, event):
        self._events.append(event)

    def poll(self):
        """Return the events seen since the last call, oldest first, and clear the queue."""
        events, self._events = self._events, []
        return events

    def get_velocity_cmd(self):
        return self.velocity_cmd

    def zero(self):
        self.velocity_cmd[:] = 0.0

    def adjust_push_force(self, delta):
        """Change the push magnitude, never below zero, and report the new value."""
        self.push_force = max(0.0, self.push_force + delta)
        print(colored(f"push force {self.push_force:.0f} N", "cyan"))
        return self.push_force

    def shutdown(self):
        pass

    @staticmethod
    def help():
        return ""


class KeyboardCommandInterface(CommandInterfaceBase):
    """Non-blocking raw-stdin keyboard input.

    The velocity command is held between key presses rather than pulsed, so the robot keeps walking
    while you are not touching anything.  Three ways to set it:

    * direction keys step it by ``lin_step`` / ``ang_step``;
    * number keys 1-N jump the forward velocity to ``speed_presets``;
    * ``v`` starts numeric entry - type ``vx``, ``vx,vy`` or ``vx,vy,wz`` and press Enter.

    The terminal is put in cbreak mode and restored on shutdown, also via ``atexit`` so a crash does
    not leave the shell unusable.
    """

    ENTRY_TIMEOUT = 8.0
    """Numeric entry is abandoned after this long without a keystroke, so a forgotten half-typed
    command cannot swallow the action keys indefinitely."""

    KEY_MAP = {
        "p": Event.PUSH,
        "c": Event.CALIBRATE,
        "u": Event.STAND_UP,
        "r": Event.START_RL,
        "f": Event.SAFE_STOP,
        "d": Event.STAND_DOWN,
        " ": Event.STOP,
        "q": Event.QUIT,
        "x": Event.ZERO_CMD,
    }

    def __init__(self, max_lin_vel=0.5, max_ang_vel=0.5, lin_step=0.1, ang_step=0.1,
                 speed_presets=(0.0, 0.1, 0.2, 0.3, 0.4), push_force=125.0,
                 push_force_step=25.0):
        super(KeyboardCommandInterface, self).__init__(max_lin_vel, max_ang_vel, push_force,
                                                       push_force_step)
        self.lin_step = lin_step
        self.ang_step = ang_step
        self.speed_presets = tuple(speed_presets)
        self._entry = None          # None when not typing, else the buffer so far
        self._entry_stamp = 0.0
        self._fd = sys.stdin.fileno()
        self._restored = True
        self._old_settings = None
        if not os.isatty(self._fd):
            raise RuntimeError("Keyboard input needs a tty; start the controller from a terminal "
                               "(lab_locosim / dock-other) or use --input joy")
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._restored = False
        atexit.register(self.shutdown)
        print(colored(self.help(self.lin_step, self.ang_step, self.speed_presets,
                                self.push_force), "cyan"))

    def shutdown(self):
        if not self._restored and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._restored = True

    def update(self):
        """Drain pending keystrokes.  One non-blocking select per control tick."""
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            if not key:
                break
            self._handle(key)
        if self._entry is not None and time.monotonic() - self._entry_stamp > self.ENTRY_TIMEOUT:
            self._entry = None
            print(colored("\rVelocity entry timed out", "yellow"), flush=True)

    # -- numeric entry ---------------------------------------------------------------------------
    def _handle_entry(self, key):
        # The emergency key outranks everything, including a half-typed command.
        if key == " ":
            self._entry = None
            self._post(Event.STOP)
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
        text = text.strip()
        if not text:
            print(colored("\rVelocity entry cancelled", "yellow"), flush=True)
            return
        try:
            values = [float(part) for part in text.split(",") if part.strip() != ""]
        except ValueError:
            print(colored(f"\rCannot read '{text}' as a velocity; expected vx[,vy[,wz]]", "red"),
                  flush=True)
            return
        if not 1 <= len(values) <= 3:
            print(colored(f"\rExpected 1 to 3 values, got {len(values)}", "red"), flush=True)
            return
        cmd = np.zeros(3)
        cmd[:len(values)] = values
        requested = cmd.copy()
        self.velocity_cmd[:] = cmd
        self._clamp()
        self._report("typed", requested)

    def _clamp(self):
        np.clip(self.velocity_cmd[:2], -self.max_lin_vel, self.max_lin_vel,
                out=self.velocity_cmd[:2])
        self.velocity_cmd[2] = float(np.clip(self.velocity_cmd[2],
                                             -self.max_ang_vel, self.max_ang_vel))

    def _report(self, source, requested=None):
        """Print the full resulting command after any change.

        Every way of setting a velocity ends up here, so the operator always sees the complete
        ``(vx, vy, wz)`` rather than having to track which component the last key press moved.
        """
        cmd = self.velocity_cmd
        text = (f"velocity command  vx={cmd[0]:+.2f} m/s  vy={cmd[1]:+.2f} m/s  "
                f"wz={cmd[2]:+.2f} rad/s   [{source}]")
        if requested is not None and not np.allclose(requested, cmd):
            print(colored(f"\r{text}  <- clamped from "
                          f"({requested[0]:+.2f}, {requested[1]:+.2f}, {requested[2]:+.2f}), "
                          f"limits {self.max_lin_vel} m/s and {self.max_ang_vel} rad/s",
                          "yellow"), flush=True)
        else:
            print(colored("\r" + text, "green"), flush=True)

    # -- normal mode -----------------------------------------------------------------------------
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

        if key.isdigit() and key != "0":
            index = int(key) - 1
            if index < len(self.speed_presets):
                self.velocity_cmd[0] = self.speed_presets[index]
                self.velocity_cmd[1] = 0.0
                self.velocity_cmd[2] = 0.0
                self._clamp()
                self._report(f"preset {key}")
                return

        # Before the case-folding lookup: 'P' has to stay distinct from 'p'.
        if key == "P":
            self._post(Event.PUSH_BURST)
            return
        if key in ("+", "="):            # same physical key on most layouts
            self.adjust_push_force(self.push_force_step)
            return
        if key == "-":
            self.adjust_push_force(-self.push_force_step)
            return

        lowered = key.lower()
        if lowered in self.KEY_MAP:
            event = self.KEY_MAP[lowered]
            if event is Event.ZERO_CMD:
                self.zero()
                self._report("zeroed")
            self._post(event)
            return

        steps = {"w": (0, self.lin_step), "s": (0, -self.lin_step),
                 "a": (1, self.lin_step), "z": (1, -self.lin_step),
                 "j": (2, self.ang_step), "l": (2, -self.ang_step)}
        if lowered not in steps:
            return
        axis, delta = steps[lowered]
        requested = self.velocity_cmd.copy()
        requested[axis] += delta
        self.velocity_cmd[axis] += delta
        self._clamp()
        self._report(f"{lowered} {delta:+.2f}", requested)

    @staticmethod
    def help(lin_step=0.1, ang_step=0.1, speed_presets=(0.0, 0.1, 0.2, 0.3, 0.4),
             push_force=125.0):
        presets = "  ".join(f"{i + 1}:{v:g}" for i, v in enumerate(speed_presets))
        return ("\nKeyboard commands\n"
                "  c  calibrate IMU bias        w/s  forward / backward "
                f"(step {lin_step:g} m/s)\n"
                "  u  stand up                  a/z  left / right\n"
                "  r  start RL policy           j/l  turn left / right "
                f"(step {ang_step:g} rad/s)\n"
                "  f  SAFE STOP (stay standing) v    type an exact velocity: vx[,vy[,wz]] Enter\n"
                "  d  stand down                x    zero the velocity command\n"
                f"  SPACE  EMERGENCY damping     forward presets  {presets}\n"
                f"  q  quit\n"
                f"\nDisturbances (simulation only)\n"
                f"  p  one random push           -/+  push force -/+ "
                f"(now {push_force:.0f} N)\n"
                f"  P  a burst of random pushes\n")


class JoyCommandInterface(CommandInterfaceBase):
    """Xbox pad over the ROS ``/joy`` topic, with edge-detected buttons.

    Axis and button indices follow the standard Linux ``xpad`` layout used by the ROS ``joy`` node.
    """

    A, B, X, Y, LB, RB, BACK, START = 0, 1, 2, 3, 4, 5, 6, 7

    BUTTON_MAP = {
        Y: Event.CALIBRATE,
        A: Event.STAND_UP,
        X: Event.START_RL,
        B: Event.STAND_DOWN,
        BACK: Event.STOP,
        START: Event.QUIT,
    }

    def __init__(self, max_lin_vel=0.5, max_ang_vel=0.5, dead_zone=0.08, start_joy_node=True,
                 push_force=125.0, push_force_step=25.0):
        super(JoyCommandInterface, self).__init__(max_lin_vel, max_ang_vel, push_force,
                                                  push_force_step)
        self.dead_zone = dead_zone
        self._msg = Joy()
        self._prev_buttons = None
        if start_joy_node:
            self._start_joy_node()
        self._sub = ros.Subscriber("/joy", Joy, self._callback, queue_size=1, tcp_nodelay=True)
        print(colored(self.help(), "cyan"))

    def _start_joy_node(self):
        try:
            subprocess.run(["rosnode", "kill", "/joy_node"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except Exception:
            pass
        try:
            subprocess.Popen(["rosrun", "joy", "joy_node"])
        except Exception as exc:
            raise RuntimeError(f"Could not start the ROS joy node: {exc}")
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
            # Left stick drives translation, right stick X drives yaw rate.  Isaac trained the
            # command in the base frame, which is what the sticks map onto directly.
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
        # Both bumpers together is the panic grip: easy to hit without looking.
        both_bumpers = (len(buttons) > max(self.LB, self.RB)
                        and buttons[self.LB] and buttons[self.RB])
        if both_bumpers:
            if self.LB in pressed or self.RB in pressed:
                self._post(Event.STOP)
        elif self.RB in pressed:
            # RB on its own is the safe stop.  Grabbing the panic grip can still close RB one
            # message before LB and emit a safe stop first; that is harmless, because the
            # emergency arrives a few milliseconds later and overrides it from any state.
            self._post(Event.SAFE_STOP)
        for index in pressed:
            if index in self.BUTTON_MAP:
                self._post(self.BUTTON_MAP[index])
        self._prev_buttons = buttons

    @staticmethod
    def help():
        return ("\nJoystick commands\n"
                "  Y  calibrate IMU bias        left stick   forward / lateral\n"
                "  A  stand up                  right stick X  turn\n"
                "  X  start RL policy\n"
                "  B  stand down                RB  SAFE STOP (stay standing)\n"
                "  BACK or LB+RB  EMERGENCY damping      START  quit\n")


def create_command_interface(kind="keyboard", cfg=None, **kwargs):
    """Build a command interface.

    Args:
        kind: ``'keyboard'``, ``'joy'`` (also ``'joystick'``/``'xbox'``) or ``'none'``.
        cfg: optional controller configuration dict (see
            :mod:`base_controllers.rl_controller_config`); the input-related keys are picked out of
            it so callers do not have to unpack them by hand.
        **kwargs: overrides applied on top of ``cfg``.
    """
    options = {}
    if cfg is not None:
        options["max_lin_vel"] = cfg["max_lin_vel_cmd"]
        options["max_ang_vel"] = cfg["max_ang_vel_cmd"]
        options["push_force"] = cfg["push_force"]
        options["push_force_step"] = cfg["push_force_step"]
        if kind == "keyboard":
            options["lin_step"] = cfg["key_lin_step"]
            options["ang_step"] = cfg["key_ang_step"]
            options["speed_presets"] = cfg["key_speed_presets"]
        elif kind in ("joy", "joystick", "xbox"):
            options["dead_zone"] = cfg["joy_dead_zone"]
    options.update(kwargs)

    if kind == "keyboard":
        return KeyboardCommandInterface(**options)
    if kind in ("joy", "joystick", "xbox"):
        return JoyCommandInterface(**options)
    if kind in ("none", None):
        options.pop("dead_zone", None)
        for key in ("lin_step", "ang_step", "speed_presets"):
            options.pop(key, None)
        return NullCommandInterface(**options)
    raise ValueError(f"Unknown command interface '{kind}'")


class NullCommandInterface(CommandInterfaceBase):
    """No operator input: for scripted runs where the caller posts its own events."""

    def __init__(self, max_lin_vel=0.5, max_ang_vel=0.5, push_force=125.0,
                 push_force_step=25.0):
        super(NullCommandInterface, self).__init__(max_lin_vel, max_ang_vel, push_force,
                                                   push_force_step)

    def update(self):
        pass

    def post(self, event):
        self._post(event)
