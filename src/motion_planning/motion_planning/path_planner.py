import numpy as np


class PathPlanner:
    """
    RRT* (Rapidly-exploring Random Tree — optimal) path planner.

    Intended for obstacle-aware Cartesian or joint-space planning.
    Implementation is deferred — this class defines the interface that
    will be filled in when RRT* integration is needed.

    Expected usage
    --------------
        pp = PathPlanner(state_space='joint')   # or 'cartesian'
        pp.set_obstacles(obstacle_list)
        pp.set_start(start_config)
        pp.set_goal(goal_config)
        path = pp.plan()   # returns list of 6-vector configs, or None
    """

    def __init__(self, state_space='joint', max_iterations=5000,
                 step_size=0.1, goal_bias=0.05, search_radius=0.5):
        """
        Args:
            state_space:     'joint' (6D joint angles) or 'cartesian' (6D pose)
            max_iterations:  maximum RRT* iterations
            step_size:       extend step size (rad for joint, m for cartesian position)
            goal_bias:       probability [0, 1] of sampling the goal directly
            search_radius:   neighbour search / rewiring radius for RRT*
        """
        self.state_space = state_space
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.search_radius = search_radius

        self._start = None
        self._goal = None
        self._obstacles = []

    def set_start(self, start_config):
        """Set the start configuration (6-vector)."""
        self._start = np.array(start_config, dtype=float)

    def set_goal(self, goal_config):
        """Set the goal configuration (6-vector)."""
        self._goal = np.array(goal_config, dtype=float)

    def set_obstacles(self, obstacle_list):
        """
        Provide obstacles for collision checking.

        Args:
            obstacle_list: list of obstacle descriptors
                           (format to be defined with the collision module)
        """
        self._obstacles = obstacle_list

    def plan(self):
        """
        Run RRT* and return a collision-free path.

        Returns:
            list of np.ndarray (each a 6-vector config) from start to goal,
            or None if planning failed.

        Note:
            NOT IMPLEMENTED — raises NotImplementedError until RRT* is integrated.
            For straight-line motion use PathInterpolator directly.
        """
        raise NotImplementedError(
            'RRT* planning is not yet implemented. '
            'Use PathInterpolator for straight-line joint/cartesian paths.'
        )

