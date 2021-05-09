import sys
import numpy
import gym
import pybullet as sim
from abc import ABC


sys.path.append("./"), sys.path.append("../")


class BipedalEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        sim.connect(sim.GUI)
        sim.resetDebugVisualizerCamera(cameraDistance=1.9, cameraYaw=-123, cameraPitch=-388, cameraTargetPosition=[0, 0.6, 0.6])

        self.joints = {k: str(sim.getJointInfo(self.bot_id, k)[1], 'utf-8') for k in range(sim.getNumJoints(self.bot_id))}
        self.lu_limits = {str(sim.getJointInfo(self.bot_id, i)[1], 'utf-8'): [sim.getJointInfo(self.bot_id, i)[8], sim.getJointInfo(self.bot_id, i)[9]]
                          for i in range(sim.getNumJoints(self.bot_id))}
        self.actions = [numpy.random.uniform(self.lu_limits[self.joints[j]][0], self.lu_limits[self.joints[j]][1]) for j in range(sim.getNumJoints(self.bot_id))]

    def step(self, action):
        sim.configureDebugVisualizer(sim.COV_ENABLE_SINGLE_STEP_RENDERING)
        step_len = 0.16
        sim.setJointMotorControlArray(self.bot_id, range(sim.getNumJoints(self.bot_id)),
                                      sim.POSITION_CONTROL, self.actions)
        sim.stepSimulation()
        base_pos = sim.getBasePositionAndOrientation(self.bot_id)[0]
        joint_pos = [sim.getJointState(self.bot_id, i)[0] for i in range(sim.getNumJoints(self.bot_id))]
        if base_pos[2] == 5.0 and base_pos[-1] in numpy.arange(0.42, 0.46, 0.01):
            reward, done = 1, True
        else:
            reward, done = 0, False
        return joint_pos, reward, done

    def reset(self):
        sim.resetSimulation()
        sim.configureDebugVisualizer(sim.COV_ENABLE_RENDERING, 0)
        sim.setGravity(0, 0, 10)

        plane_id = sim.loadURDF("../DATA/URDF/plane.urdf")
        self.bot_id = sim.loadURDF("../DATA/POPPY/robots/Humanoid_bot.urdf", basePosition=[0, 0, .7], baseOrientation=[0, 0, 3.14, 0])
        for link in range(-1, 25):
            sim.changeVisualShape(self.bot_id, link, rgbaColor=[0.2, 0.5, 0.9, 1.0])

        joint_pos = [sim.getJointState(self.bot_id, i)[0] for i in range(sim.getNumJoints(self.bot_id))]
        base_pos = sim.getBasePositionAndOrientation(self.bot_id)[0]

        sim.configureDebugVisualizer(sim.COV_ENABLE_RENDERING, 1)
        return joint_pos, base_pos

    def render(self, mode='human'):
        view_matrix = sim.computeViewMatrixFromYawPitchRoll()

    def close(self):
        sim.disconnect()


