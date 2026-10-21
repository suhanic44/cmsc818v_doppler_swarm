
import pybullet as p
import pybullet_data
import numpy as np


class SimulationWorld:
    def __init__(self, gui=True, gravity=False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)

        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")

        self.dt = 1.0 / 240.0

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.client
        )

        p.setGravity(
            0, 0, -9.81 if gravity else 0,
            physicsClientId=self.client
        )

        p.setTimeStep(
            self.dt,
            physicsClientId=self.client
        )

        p.loadURDF(
            "plane.urdf",
            physicsClientId=self.client
        )

        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=10,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=[0, 0, 2],
                physicsClientId=self.client
            )

    def create_uav(self, position):
        """Create a stationary UAV represented by a blue box."""

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.1],
            physicsClientId=self.client
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.1],
            rgbaColor=[0.1, 0.3, 0.9, 1],
            physicsClientId=self.client
        )

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self.client
        )

    def create_projectile(
        self,
        position,
        velocity=(0, 0, 0),
        radius=0.1,
        mass=0.1
    ):
        """Create a moving spherical projectile."""

        collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
            physicsClientId=self.client
        )

        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.9, 0.1, 0.1, 1],
            physicsClientId=self.client
        )

        projectile_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self.client
        )

        p.changeDynamics(
            projectile_id,
            -1,
            linearDamping=0.0,
            angularDamping=0.0,
            physicsClientId=self.client
        )

        p.resetBaseVelocity(
            projectile_id,
            linearVelocity=velocity,
            physicsClientId=self.client
        )

        return projectile_id

    def get_projectile_state(self, projectile_id):
        """Return projectile position and velocity."""

        position, _ = p.getBasePositionAndOrientation(
            projectile_id,
            physicsClientId=self.client
        )

        velocity, _ = p.getBaseVelocity(
            projectile_id,
            physicsClientId=self.client
        )

        return (
            np.array(position),
            np.array(velocity)
        )

    def step(self):
        p.stepSimulation(physicsClientId=self.client)

    def close(self):
        p.disconnect(self.client)
