import pybullet as p
import pybullet_data
import time
import numpy as np

gui = True
dt = 1 / 240
max_steps = 240 * 10
g = 9.81

p.connect(p.GUI if gui else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)

pend = p.loadURDF("two-link.urdf.xml", useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

joint_inds     = []
link_eef_index = None
for ji in range(p.getNumJoints(pend)):
    info = p.getJointInfo(pend, ji)
    j_name = info[1].decode()
    child_link = info[12].decode()
    if info[2] == p.JOINT_REVOLUTE:
        joint_inds.append(ji)
    if child_link == "link_eef":
        link_eef_index = ji

for ji in joint_inds:
    p.setJointMotorControl2(pend, ji, controlMode=p.VELOCITY_CONTROL, force=0)


# initial pos for joints
init_joint0 = 0.2
init_joint1 = -0.1

p.resetJointState(pend, joint_inds[0], targetValue=init_joint0)
p.resetJointState(pend, joint_inds[1], targetValue=init_joint1)


# target position
target_pos_x = 0.8
target_pos_z = 1.2

target_pos = np.array([target_pos_x, 0.0, target_pos_z])

# prop coef
kp_cart_x = 100
kp_cart_y = 100
kp_cart_z = 100

Kp_cart = np.diag([kp_cart_x, kp_cart_y, kp_cart_z])

for x in range(max_steps):
    q = np.array([p.getJointState(pend, ji)[0] for ji in joint_inds])
    qdot = np.array([p.getJointState(pend, ji)[1] for ji in joint_inds])

    ee = p.getLinkState(pend, link_eef_index, computeForwardKinematics=True)
    ee_pos = np.array(ee[4])
    err = target_pos - ee_pos
    err[1] = 0 # moving only in Oxz

    J_lin, _ = p.calculateJacobian(
        bodyUniqueId = pend,
        linkIndex = link_eef_index,
        localPosition = [0, 0, 0],
        objPositions = list(q),
        objVelocities = list(qdot),
        objAccelerations = [0.0] * len(q)
    )
    J = np.array(J_lin)
    J_pinv = np.linalg.pinv(J)
    v_cart = Kp_cart @ err
    qdot_des = J_pinv @ v_cart

    for idx, ji in enumerate(joint_inds):
        p.setJointMotorControl2(
            bodyIndex = pend,
            jointIndex = ji,
            controlMode = p.VELOCITY_CONTROL,
            targetVelocity = qdot_des[idx],
            force = 50
        )

    p.stepSimulation()
    if gui:
        time.sleep(dt)

p.disconnect()
