import numpy as np


class TrajectoryPlanner:
    """
    Time-parameterized trajectory planner for multi-joint robots.

    Two modes:
      1. Trapezoidal velocity profile  — init_trajectory_planner(start_vels, path_lengths, v_max, a_max)
         Computes time-optimal trapezoidal profile for each joint independently,
         then scales all joints to finish at the same time (synchronised motion).

      2. Cubic polynomial (fixed time)  — init_trajectory_planner_time(path_lengths, total_time)
         Fits a cubic polynomial that starts and ends at zero velocity.

    After initialisation call get_trajectory_points() to obtain sampled waypoints.
    """

    def __init__(self, dt=0.01):
        self.dt = dt
        self.path_lengths = None
        self.total_time = None
        self._mode = None          # 'trapezoid' or 'cubic'
        self._v_max = None
        self._a_max = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_trajectory_planner(self, start_velocities, path_lengths, v_max, a_max):
        """
        Initialise a synchronised trapezoidal trajectory.

        Args:
            start_velocities: array of starting velocities (typically all zeros)
            path_lengths:     signed displacement per joint (from PathInterpolator)
            v_max:            scalar peak velocity (rad/s)
            a_max:            scalar peak acceleration (rad/s²)

        Returns:
            total_time (float, seconds)
        """
        self.path_lengths = np.array(path_lengths, dtype=float)
        self._v_max = float(v_max)
        self._a_max = float(a_max)
        self._mode = 'trapezoid'

        abs_lengths = np.abs(self.path_lengths)
        t_acc = v_max / a_max
        d_acc = 0.5 * a_max * t_acc ** 2

        joint_times = []
        for L in abs_lengths:
            if L < 1e-9:
                joint_times.append(0.0)
            elif 2 * d_acc >= L:
                # Triangular profile — cannot reach v_max
                joint_times.append(2.0 * np.sqrt(L / a_max))
            else:
                # Trapezoidal
                joint_times.append(2 * t_acc + (L - 2 * d_acc) / v_max)

        self.total_time = max(joint_times) if joint_times else 0.0
        return self.total_time

    def init_trajectory_planner_time(self, path_lengths, total_time):
        """
        Initialise a cubic-polynomial trajectory with a fixed duration.

        Args:
            path_lengths: signed displacement per joint
            total_time:   desired motion duration (seconds)

        Returns:
            total_time (float)
        """
        self.path_lengths = np.array(path_lengths, dtype=float)
        self.total_time = float(total_time)
        self._mode = 'cubic'
        return self.total_time

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def get_trajectory_points(self):
        """
        Sample the trajectory at every dt interval.

        Returns:
            list of dicts: {'time': float, 'positions': np.ndarray, 'velocities': np.ndarray}
            where positions/velocities are *signed displacements from start*
            (pass to PathInterpolator.get_joint_position_at to get absolute angles).
        """
        if self.path_lengths is None:
            raise RuntimeError('TrajectoryPlanner: call init_trajectory_planner*() first.')

        if self.total_time < 1e-9:
            # Zero-length motion
            n = len(self.path_lengths)
            return [{'time': 0.0,
                     'positions': np.zeros(n),
                     'velocities': np.zeros(n)}]

        times = np.arange(0.0, self.total_time + self.dt, self.dt)
        # Clamp last point to exactly total_time
        times[-1] = min(times[-1], self.total_time)

        points = []
        for t in times:
            if self._mode == 'trapezoid':
                pos, vel = self._trapezoid_state(t)
            else:
                pos, vel = self._cubic_state(t)
            points.append({'time': float(t), 'positions': pos, 'velocities': vel})
        return points

    # ------------------------------------------------------------------
    # Internal profile functions
    # ------------------------------------------------------------------

    def _trapezoid_state(self, t):
        """
        Synchronised trapezoidal profile.
        Each joint is scaled so that it uses the full total_time at reduced velocity.
        """
        T = self.total_time
        pos = np.zeros(len(self.path_lengths))
        vel = np.zeros(len(self.path_lengths))

        for i, L_signed in enumerate(self.path_lengths):
            L = abs(L_signed)
            sign = np.sign(L_signed)
            if L < 1e-9:
                continue

            # Scale v_max / a_max so this joint takes exactly total_time
            # Using the synchronisation approach: find effective v and a
            # that produce a trapezoidal profile of duration T over distance L.
            # Solve: T = v/a + L/v  =>  v^2 - T*a*v + L*a = 0
            a_eff = self._a_max
            discriminant = (T * a_eff) ** 2 - 4 * a_eff * L
            if discriminant < 0:
                # Force triangular: a_eff = 4L/T^2
                a_eff = 4.0 * L / (T ** 2)
                v_eff = a_eff * T / 2.0
                t_acc = T / 2.0
                t_flat_end = T / 2.0
            else:
                v_eff = (T * a_eff - np.sqrt(discriminant)) / 2.0
                v_eff = min(v_eff, self._v_max)
                if v_eff < 1e-12:
                    continue
                t_acc = v_eff / a_eff
                t_flat_end = T - t_acc

            t_clamped = min(t, T)
            p, v = self._trapezoid_single(t_clamped, L, v_eff, a_eff, t_acc, t_flat_end)
            pos[i] = sign * p
            vel[i] = sign * v

        return pos, vel

    def _trapezoid_single(self, t, L, v_eff, a_eff, t_acc, t_flat_end):
        if t <= t_acc:
            p = 0.5 * a_eff * t ** 2
            v = a_eff * t
        elif t <= t_flat_end:
            d_acc = 0.5 * a_eff * t_acc ** 2
            dt = t - t_acc
            p = d_acc + v_eff * dt
            v = v_eff
        else:
            d_acc = 0.5 * a_eff * t_acc ** 2
            d_flat = v_eff * (t_flat_end - t_acc)
            dt = t - t_flat_end
            p = d_acc + d_flat + v_eff * dt - 0.5 * a_eff * dt ** 2
            v = v_eff - a_eff * dt
        p = np.clip(p, 0.0, L)
        v = max(0.0, v)
        return p, v

    def _cubic_state(self, t):
        """Cubic polynomial: zero velocity at start and end."""
        T = self.total_time
        s = np.clip(t / T, 0.0, 1.0)
        # Cubic Hermite: p(s) = L*(3s^2 - 2s^3),  v(s) = L*(6s - 6s^2)/T
        pos = self.path_lengths * (3 * s ** 2 - 2 * s ** 3)
        vel = self.path_lengths * (6 * s - 6 * s ** 2) / T
        return pos, vel
