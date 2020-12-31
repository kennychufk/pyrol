from datetime import timedelta

from ._pyrol import mdl, sg, plan, math
import numpy as np


class RobotKinematics:
    def __init__(self, rlmdl_path, rlsg_path):
        self.mdl_kinematic = mdl.Kinematic()
        self.scene = sg.bullet.Scene()
        self.planner = plan.Prm()
        self.plan_simple_model = plan.SimpleModel()
        self.nn = plan.KdtreeNearestNeighbors(self.plan_simple_model)
        self.sampler = plan.UniformSampler()
        self.verifier = plan.RecursiveVerifier()
        self.optimizer = plan.SimpleOptimizer()

        mdl_factory = mdl.XmlFactory()
        mdl_factory.load(rlmdl_path, self.mdl_kinematic)
        sg_factory = sg.XmlFactory()
        sg_factory.load(rlsg_path, self.scene)

        self.ik = mdl.NloptInverseKinematics(self.mdl_kinematic)
        # self.ik.duration = timedelta(milliseconds=100)

        self.planner.model = self.plan_simple_model
        self.plan_simple_model.mdl = self.mdl_kinematic
        self.plan_simple_model.model = self.scene.getModel(0)
        self.plan_simple_model.scene = self.scene

        self.sampler.model = self.plan_simple_model
        self.planner.sampler = self.sampler

        self.verifier.delta = np.radians(1)
        self.verifier.model = self.plan_simple_model
        self.planner.verifier = self.verifier
        # self.planner.duration = timedelta(seconds=0.1)

        self.planner.setNearestNeighbors(self.nn)

        self.optimizer.model = self.plan_simple_model
        self.optimizer.verifier = self.verifier

        self.mdl_kinematic.setPosition(self.mdl_kinematic.home)
        self.mdl_kinematic.forwardPosition()

    @staticmethod
    def decode_transform_controls(translation_mm, rpy_deg):
        translation = np.array(translation_mm) / 1000.0
        ypr_angles = np.radians(np.flip(rpy_deg))
        rotation = math.Quaternion.fromEulerAngles(ypr_angles)
        transform = math.Transform.Identity()
        transform.translate(translation)
        transform.rotate(rotation)
        return transform

    def forward_angles(self, angles):
        self.mdl_kinematic.setPosition(angles)
        self.mdl_kinematic.forwardPosition()
        return self.mdl_kinematic.getOperationalPosition(0)

    def check_collision_free(self, angles):
        self.planner.start = angles
        self.planner.goal = angles
        return self.planner.verify()

    def inverse_transform(self, transform):
        self.ik.addGoal(transform, 0)
        if (self.ik.solve()):
            return self.mdl_kinematic.getPosition()
        return None

    def get_interpolated_path(self, start_angles, goal_angles, num_steps):
        path = np.zeros([num_steps, 6])
        for i in range(num_steps):
            path[i] = self.plan_simple_model.interpolate(
                start_angles, goal_angles, (i + 1) / (num_steps + 1))
        return path

    def plan_angles(self, start_angles, goal_angles):
        self.mdl_kinematic.setPosition(start_angles)
        self.mdl_kinematic.forwardPosition()
        self.planner.start = self.mdl_kinematic.getPosition()
        self.planner.goal = goal_angles
        if (not self.planner.verify()):
            return
        self.planner.solve()
        path = self.planner.getPath()
        self.optimizer.process(path)
        return path
