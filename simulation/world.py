
import os

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

        # Resolve assets relative to the project root,
        # regardless of where Python is launched.
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.drone_mesh_path = os.path.join(
            self.project_root,
            "assets",
            "quadcopter.obj"
        )

        if gui:
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI,
                0,
                physicsClientId=self.client
            )

            p.resetDebugVisualizerCamera(
                cameraDistance=4.5,
                cameraYaw=35,
                cameraPitch=-35,
                cameraTargetPosition=[0.5, 0.5, 0.5],
                physicsClientId=self.client
            )

    def create_uav(
        self,
        position,
        radius=None,
        mesh_scale=0.1
    ):
        """
        Create a UAV.

        If radius is None, use the original blue box.
        Otherwise, use a spherical collision shape.

        When assets/quadcopter.obj exists, display the
        drone mesh instead of the sphere.

        mesh_scale controls visual size only.
        """

        if radius is None:
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

        else:
            if radius <= 0:
                raise ValueError("UAV radius must be positive")

            collision = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=radius,
                physicsClientId=self.client
            )

            if os.path.isfile(self.drone_mesh_path):
                visual = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=self.drone_mesh_path,
                    meshScale=[
                        mesh_scale,
                        mesh_scale,
                        mesh_scale
                    ],
                    rgbaColor=[0.2, 0.4, 0.9, 1],
                    physicsClientId=self.client
                )
            else:
                visual = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=radius,
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

    def get_uav_position(self, uav_id):
        """Return a UAV's current position."""

        position, _ = p.getBasePositionAndOrientation(
            uav_id,
            physicsClientId=self.client
        )

        return np.array(position, dtype=float)

    def get_uav_positions(self, uav_ids):
        """Return all UAV positions."""

        return np.array([
            self.get_uav_position(uav_id)
            for uav_id in uav_ids
        ], dtype=float)

    def set_uav_position(self, uav_id, position):
        """Move a UAV kinematically."""

        position = np.asarray(position, dtype=float)

        if position.shape != (3,):
            raise ValueError(
                "UAV position must have shape (3,)"
            )

        p.resetBasePositionAndOrientation(
            uav_id,
            position.tolist(),
            [0, 0, 0, 1],
            physicsClientId=self.client
        )

    def set_uav_positions(self, uav_ids, positions):
        """Move multiple UAVs to specified positions."""

        positions = np.asarray(positions, dtype=float)

        if positions.shape != (len(uav_ids), 3):
            raise ValueError(
                "UAV positions must have shape (N, 3)"
            )

        for uav_id, position in zip(uav_ids, positions):
            self.set_uav_position(uav_id, position)

    def execute_uav_movement(
        self,
        uav_ids,
        initial_positions,
        displacements,
        elapsed_time,
        maneuver_duration
    ):
        """
        Execute straight-line SQP movement.

        s_i(t) = s_i(0) + alpha(t) * displacement_i
        alpha(t) = clip(t / duration, 0, 1)
        """

        if maneuver_duration <= 0:
            raise ValueError(
                "Maneuver duration must be positive"
            )

        initial_positions = np.asarray(
            initial_positions,
            dtype=float
        )

        displacements = np.asarray(
            displacements,
            dtype=float
        )

        if initial_positions.shape != (len(uav_ids), 3):
            raise ValueError(
                "Initial positions must have shape (N, 3)"
            )

        if displacements.shape != initial_positions.shape:
            raise ValueError(
                "Displacements must match initial positions"
            )

        alpha = np.clip(
            elapsed_time / maneuver_duration,
            0.0,
            1.0
        )

        positions = (
            initial_positions + alpha * displacements
        )

        self.set_uav_positions(uav_ids, positions)

        return positions

    def step(self):
        p.stepSimulation(physicsClientId=self.client)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
