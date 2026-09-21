"""
pid_controller.py
-------------------
A reusable PID controller, using the values you tuned yourself in
pid_simulation.py. This keeps its internal memory (integral,
previous_error) between calls, since a servo tracking a moving object
needs to remember its state across multiple photo cycles.
"""


class PIDController:
    def __init__(self, Kp=0.4, Ki=0.002, Kd=0.3, max_correction=15):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_correction = max_correction  # safety cap: max degrees per cycle
        self.integral = 0
        self.previous_error = 0

    def step(self, error: float) -> float:
        """
        Call this once per cycle with the current error.
        Returns the correction to apply (clamped for safety).
        """
        self.integral += error
        derivative = error - self.previous_error

        correction = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        # Safety clamp: never move the servo too far in a single cycle
        correction = max(-self.max_correction, min(self.max_correction, correction))

        self.previous_error = error
        return correction

    def reset(self):
        """Call this if tracking is lost / restarted, to clear old memory."""
        self.integral = 0
        self.previous_error = 0