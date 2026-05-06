import numpy as np
import pinocchio as pin
from ur3e_vision_pick_place.inverse_kinematics import load_pinocchio

model, data, ee_frame_id = load_pinocchio("/tmp/ur3e.urdf")
print("EE frame ID:", ee_frame_id)
print("EE frame name:", model.frames[ee_frame_id].name)
print("nq:", model.nq)

# FK at home position [0, -1.57, 1.57, -1.57, -1.57, 0]
q_home = np.array([0, -1.57, 1.57, -1.57, -1.57, 0, 0])  # 7 if nq=7
pin.forwardKinematics(model, data, q_home)
pin.updateFramePlacements(model, data)
print("EE pos at home:", data.oMf[ee_frame_id].translation)
print("Valid frame?", ee_frame_id < len(model.frames))