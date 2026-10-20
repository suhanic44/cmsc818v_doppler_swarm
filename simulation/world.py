
import pybullet as p
import pybullet_data


class SimulationWorld:
    def __init__(self, gui=True):
        self.client = p.connect(p.GUI if gui else p.DIRECT)

        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)

        p.loadURDF("plane.urdf")

        p.resetDebugVisualizerCamera(
            cameraDistance=10,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 2]
        )

    def create_uav(self, position):
        """Create a simple hovering UAV represented by a blue box."""
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.1]
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.1],
            rgbaColor=[0.1, 0.3, 0.9, 1]
        )

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position
        )

    def create_projectile(self, position, radius=0.1):
        """Create a red spherical projectile."""
        collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius
        )

        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.9, 0.1, 0.1, 1]
        )

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position
        )

    def step(self):
        p.stepSimulation()

    def close(self):
        p.disconnect(self.client)
