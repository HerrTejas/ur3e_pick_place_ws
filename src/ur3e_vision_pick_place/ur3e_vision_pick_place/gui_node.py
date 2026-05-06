#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

import tkinter as tk
from tkinter import ttk
import threading
import signal
import sys

UR_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]


class ControlUI(Node):
    def __init__(self):
        super().__init__('gui_node')

        self.joint_pub = self.create_publisher(
            Float64MultiArray, '/cmd_joint_positions', 10)

        self.pose_pub = self.create_publisher(
            PoseStamped, '/path_target_pose', 10)

        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(PoseStamped, '/end_effector_pose', self.pose_cb, 10)

        self.current_joints = [0.0] * 6
        self.current_pose = None

    def joint_cb(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        try:
            self.current_joints = [
                msg.position[name_to_idx[n]] for n in UR_JOINT_NAMES
            ]
        except:
            pass

    def pose_cb(self, msg):
        self.current_pose = msg

    def publish_joints(self, joints):
        msg = Float64MultiArray()
        msg.data = joints
        self.joint_pub.publish(msg)

    def publish_pose(self, x, y, z, qx, qy, qz, qw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.pose_pub.publish(msg)


class App:
    def __init__(self, root, node):
        self.root = root
        self.node = node

        root.title("UR3e Control Panel")
        root.geometry("520x780")

        style = ttk.Style()
        style.configure("TButton", padding=6)
        style.configure("TLabel", font=("Arial", 10))
        style.configure("Status.TLabel", font=("Arial", 10, "bold"), foreground="blue")

        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # -------- End Effector Status Section --------
        status_frame = ttk.LabelFrame(main_frame, text="End Effector Status (Live)")
        status_frame.pack(fill="x", pady=6)

        self.status_vars = {}
        status_labels = ["X", "Y", "Z", "Qx", "Qy", "Qz", "Qw"]

        for i, label in enumerate(status_labels):
            ttk.Label(status_frame, text=label, width=5).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value="—")
            ttk.Label(status_frame, textvariable=var, width=16,
                      relief="sunken", anchor="e",
                      font=("Courier", 10)).grid(row=i, column=1, padx=5, pady=2, sticky="w")
            self.status_vars[label] = var

        ttk.Button(status_frame, text="Refresh Now",
                   command=self.refresh_status).grid(row=len(status_labels), column=0,
                                                     columnspan=2, pady=6)

        # -------- Joint Section --------
        joint_frame = ttk.LabelFrame(main_frame, text="Joint Control")
        joint_frame.pack(fill="x", pady=6)

        self.joint_entries = []
        for i, name in enumerate(UR_JOINT_NAMES):
            ttk.Label(joint_frame, text=name).grid(row=i, column=0, sticky="w")
            e = ttk.Entry(joint_frame, width=12)
            e.grid(row=i, column=1, padx=5, pady=2)
            self.joint_entries.append(e)

        ttk.Button(joint_frame, text="Update from Robot",
                   command=self.update_joint).grid(row=6, column=0, pady=5)
        ttk.Button(joint_frame, text="Send",
                   command=self.send_joint).grid(row=6, column=1)

        # -------- Cartesian Section (Quaternion) --------
        pose_frame = ttk.LabelFrame(main_frame, text="Cartesian Control (Quaternion)")
        pose_frame.pack(fill="x", pady=6)

        self.pose_entries = {}
        pose_labels = ["X", "Y", "Z", "Qx", "Qy", "Qz", "Qw"]

        for i, l in enumerate(pose_labels):
            ttk.Label(pose_frame, text=l).grid(row=i, column=0, padx=5)
            e = ttk.Entry(pose_frame, width=12)
            e.grid(row=i, column=1, padx=5, pady=2)
            self.pose_entries[l] = e

        # Default unit quaternion
        self.pose_entries["Qw"].insert(0, "1.0")
        for k in ["Qx", "Qy", "Qz"]:
            self.pose_entries[k].insert(0, "0.0")

        ttk.Button(pose_frame, text="Update from Robot",
                   command=self.update_pose).grid(row=7, column=0, pady=5)
        ttk.Button(pose_frame, text="Send",
                   command=self.send_pose).grid(row=7, column=1)

        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Status: Ready", style="Status.TLabel")
        self.status_bar.pack(pady=8)

        root.protocol("WM_DELETE_WINDOW", self.shutdown)

        # Auto-refresh status every 200 ms
        self._auto_refresh()

    def _auto_refresh(self):
        self.refresh_status()
        self.root.after(200, self._auto_refresh)

    def refresh_status(self):
        pose = self.node.current_pose
        if pose is None:
            for var in self.status_vars.values():
                var.set("—")
            return

        p = pose.pose
        values = {
            "X":  p.position.x,
            "Y":  p.position.y,
            "Z":  p.position.z,
            "Qx": p.orientation.x,
            "Qy": p.orientation.y,
            "Qz": p.orientation.z,
            "Qw": p.orientation.w,
        }
        for key, val in values.items():
            self.status_vars[key].set(f"{val:.6f}")

    def update_joint(self):
        for i in range(6):
            self.joint_entries[i].delete(0, tk.END)
            self.joint_entries[i].insert(0, f"{self.node.current_joints[i]:.4f}")
        self.status_bar.config(text="Joint values updated from robot")

    def send_joint(self):
        try:
            vals = [float(e.get()) for e in self.joint_entries]
            self.node.publish_joints(vals)
            self.status_bar.config(text="✓ Joint command sent")
        except Exception as ex:
            self.status_bar.config(text=f"Invalid joint input: {ex}")

    def update_pose(self):
        if self.node.current_pose is None:
            self.status_bar.config(text="No pose received yet")
            return

        p = self.node.current_pose.pose

        for key, val in zip(["X", "Y", "Z"],
                            [p.position.x, p.position.y, p.position.z]):
            self.pose_entries[key].delete(0, tk.END)
            self.pose_entries[key].insert(0, f"{val:.6f}")

        for key, val in zip(["Qx", "Qy", "Qz", "Qw"],
                            [p.orientation.x, p.orientation.y,
                             p.orientation.z, p.orientation.w]):
            self.pose_entries[key].delete(0, tk.END)
            self.pose_entries[key].insert(0, f"{val:.6f}")

        self.status_bar.config(text="Cartesian values updated from robot")

    def send_pose(self):
        try:
            x  = float(self.pose_entries["X"].get())
            y  = float(self.pose_entries["Y"].get())
            z  = float(self.pose_entries["Z"].get())
            qx = float(self.pose_entries["Qx"].get())
            qy = float(self.pose_entries["Qy"].get())
            qz = float(self.pose_entries["Qz"].get())
            qw = float(self.pose_entries["Qw"].get())

            self.node.publish_pose(x, y, z, qx, qy, qz, qw)
            self.status_bar.config(text="✓ Pose command sent")

        except Exception as ex:
            self.status_bar.config(text=f"Invalid pose input: {ex}")

    def shutdown(self):
        self.status_bar.config(text="Shutting down...")
        self.root.quit()
        rclpy.shutdown()
        sys.exit(0)


def main():
    rclpy.init()
    node = ControlUI()

    def spin():
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()

    root = tk.Tk()
    app = App(root, node)

    signal.signal(signal.SIGINT, lambda sig, frame: app.shutdown())

    root.mainloop()


if __name__ == "__main__":
    main()