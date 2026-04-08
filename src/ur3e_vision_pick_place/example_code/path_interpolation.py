import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
from trajectory_profile import TrajectoryProfile

class PathInterpolation:
    def __init__(self, start_pose, end_pose):
        self.start_pose = np.array(start_pose)
        self.end_pose = np.array(end_pose)

        self.path_length = np.linalg.norm(self.end_pose[:3] - self.start_pose[:3])

        self.r1, self.r2 = self.euler_to_quat()

        self.orientation_length = self.compute_orientation_distance()

    def euler_to_quat(self):
        r1 = R.from_euler('xyz', self.start_pose[3:], degrees=True)
        r2 = R.from_euler('xyz', self.end_pose[3:], degrees=True)
        return r1, r2

    def compute_orientation_distance(self):
        r_rel = self.r1.inv() * self.r2

        angle = r_rel.magnitude()
        return angle

    def interpolate(self, t):

        pos = (1 - t) * self.start_pose[:3] + t * self.end_pose[:3]

        times = [0, 1]
        rots = R.concatenate([self.r1, self.r2])
        slerp = Slerp(times, rots)

        r_interp = slerp(t)
        euler_interp = r_interp.as_euler('xyz', degrees=True)

        return np.concatenate([pos, euler_interp])
    

class PathGeneration():
    def __init__(self):
        self.trajectory = TrajectoryProfile()

    def generate_synced_trajectory(self,path, vmax=0.5, amax=0.5, dt=0.01):

        L_pos = path.path_length
        L_ori = path.orientation_length

        L_max = max(L_pos, L_ori)

        t_list, s_all, T = self.trajectory.trapezoid_multi(
            [L_pos, L_ori], vmax, amax, dt
        )

        s_pos = s_all[0]
        s_ori = s_all[1]

        trajectory = []

        for i in range(len(t_list)):
            t_pos = s_pos[i] / L_pos if L_pos > 1e-6 else 0
            t_ori = s_ori[i] / L_ori if L_ori > 1e-6 else 0

            t_interp = max(t_pos, t_ori)

            pose = path.interpolate(t_interp)

            trajectory.append(pose)

        return np.array(trajectory)

    def plot_profiles(self,path, vmax=0.5, amax=0.5, dt=0.01):

        L_pos = path.path_length
        L_ori = path.orientation_length

        L_max = max(L_pos, L_ori)

        t_list, s_base, T = self.trajectory.trapezoid_time_scaled(L_max, vmax, amax, dt)

        # Scale
        s_pos = s_base * (L_pos / L_max)
        s_ori = s_base * (L_ori / L_max)

        # Velocity (numerical derivative)
        v_pos = np.gradient(s_pos, dt)
        v_ori = np.gradient(s_ori, dt)

        # -------- Plot s(t)
        plt.figure()
        plt.plot(t_list, s_pos, label="Position s(t)")
        plt.plot(t_list, s_ori, label="Orientation s(t)")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance")
        plt.title("Position & Orientation Distance vs Time")
        plt.legend()
        plt.grid()

        # -------- Plot v(t)
        plt.figure()
        plt.plot(t_list, v_pos, label="Position v(t)")
        plt.plot(t_list, v_ori, label="Orientation v(t)")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity")
        plt.title("Position & Orientation Velocity vs Time")
        plt.legend()
        plt.grid()

        plt.show()

def main():
    start = [0, 0, 0, 0, 0, 0]
    end   = [1, 1, 1, 90, 0, 0]

    generate_path  = PathGeneration()
    path = PathInterpolation(start, end)

    print("Position distance:", path.path_length)
    print("Orientation distance (rad):", path.orientation_length)

    traj = generate_path.generate_synced_trajectory(path, vmax=0.5, amax=0.5, dt=0.01)

    print("Trajectory size:", len(traj))
    generate_path.plot_profiles(path)

if __name__ == "__main__":
    main()