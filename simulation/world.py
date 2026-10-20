
import time
import pybullet as p
import pybullet_data

def main():
    # Open PyBullet's graphical interface
    client = p.connect(p.GUI)

    if client < 0:
        raise RuntimeError("Could not connect to PyBullet GUI")

    # Load built-in models
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Add a ground plane
    p.loadURDF("plane.urdf")

    # Add a simple box
    p.loadURDF(
        "cube_small.urdf",
        basePosition=[0, 0, 1]
    )

    # Set camera position
    p.resetDebugVisualizerCamera(
        cameraDistance=3,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )

    # Run simulation for 10 seconds
    for _ in range(2400):
        p.stepSimulation()
        time.sleep(1 / 240)

    p.disconnect()


if __name__ == "__main__":
    main()
