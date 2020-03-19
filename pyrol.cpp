#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <rl/mdl/InverseKinematics.h>
#include <rl/mdl/JacobianInverseKinematics.h>
#include <rl/mdl/Kinematic.h>
#include <rl/mdl/Model.h>
#include <rl/mdl/NloptInverseKinematics.h>
#include <rl/mdl/XmlFactory.h>

#include <rl/mdl/Body.h>
#include <rl/mdl/Compound.h>
#include <rl/mdl/Frame.h>
#include <rl/mdl/Joint.h>
#include <rl/mdl/Transform.h>
#include <rl/mdl/World.h>

#include <rl/plan/GnatNearestNeighbors.h>
#include <rl/plan/KdtreeBoundingBoxNearestNeighbors.h>
#include <rl/plan/KdtreeNearestNeighbors.h>
#include <rl/plan/LinearNearestNeighbors.h>
#include <rl/plan/NearestNeighbors.h>

#include <rl/plan/RecursiveVerifier.h>
#include <rl/plan/SequentialVerifier.h>
#include <rl/plan/Verifier.h>

#include <rl/plan/BridgeSampler.h>
#include <rl/plan/GaussianSampler.h>
#include <rl/plan/Sampler.h>
#include <rl/plan/UniformSampler.h>

#include <rl/plan/Planner.h>
#include <rl/plan/Prm.h>
#include <rl/plan/Rrt.h>

#include <rl/plan/SimpleModel.h>
#include <rl/plan/Viewer.h>

#include <rl/sg/Body.h>
#include <rl/sg/Model.h>
#include <rl/sg/Shape.h>
#include <rl/sg/XmlFactory.h>
#include <rl/sg/bullet/Body.h>
#include <rl/sg/bullet/Model.h>
#include <rl/sg/bullet/Scene.h>
#include <rl/sg/bullet/Shape.h>

#include <sstream>
#include <utility>

namespace rl {
namespace mdl {
class PyInverseKinematics : public InverseKinematics {
 public:
  using InverseKinematics::InverseKinematics;
  virtual bool solve() {
    PYBIND11_OVERLOAD_PURE(bool, InverseKinematics, solve);
  }
};
class PyIterativeInverseKinematics : public IterativeInverseKinematics {
 public:
  using IterativeInverseKinematics::IterativeInverseKinematics;
  virtual bool solve() {
    PYBIND11_OVERLOAD_PURE(bool, IterativeInverseKinematics, solve);
  }
};
}  // namespace mdl
namespace sg {
class PyShape : public Shape {
 public:
  using Shape::Shape;
  virtual void getTransform(::rl::math::Transform& transform) override {
    PYBIND11_OVERLOAD_PURE(void, Shape, getTransform, transform);
  }
  virtual void setTransform(const ::rl::math::Transform& transform) override {
    PYBIND11_OVERLOAD_PURE(void, Shape, setTransform, transform);
  }
};
class PyBody : public Body {
 public:
  using Body::Body;
  virtual Shape* create(::SoVRMLShape* shape) override {
    PYBIND11_OVERLOAD_PURE(Shape*, Body, create, shape);
  }
  virtual void getFrame(::rl::math::Transform& frame) override {
    PYBIND11_OVERLOAD_PURE(void, Body, getFrame, frame);
  }
  virtual void setFrame(const ::rl::math::Transform& frame) override {
    PYBIND11_OVERLOAD_PURE(void, Body, setFrame, frame);
  }
};
class PyModel : public Model {
 public:
  using Model::Model;
  virtual Body* create() override {
    PYBIND11_OVERLOAD_PURE(Body*, Model, create);
  }
};
class PyScene : public Scene {
 public:
  using Scene::Scene;
  virtual Model* create() override {
    PYBIND11_OVERLOAD_PURE(Model*, Scene, create);
  }
};
}  // namespace sg
namespace plan {
class PyNearestNeighbors : public NearestNeighbors {
 public:
  using NearestNeighbors::NearestNeighbors;
  virtual void clear() override {
    PYBIND11_OVERLOAD_PURE(void, NearestNeighbors, clear);
  }
  virtual bool empty() const override {
    PYBIND11_OVERLOAD_PURE(bool, NearestNeighbors, empty);
  }
  virtual ::std::vector<Neighbor> nearest(
      const Value& query, const ::std::size_t& k,
      const bool& sorted = true) const override {
    PYBIND11_OVERLOAD_PURE(::std::vector<Neighbor>, NearestNeighbors, nearest,
                           query, k, sorted);
  }
  virtual void push(const Value& value) override {
    PYBIND11_OVERLOAD_PURE(void, NearestNeighbors, push, value);
  }
  virtual ::std::vector<Neighbor> radius(
      const Value& query, const Distance& radius,
      const bool& sorted = true) const override {
    PYBIND11_OVERLOAD_PURE(::std::vector<Neighbor>, NearestNeighbors, radius,
                           query, radius, sorted);
  }
  virtual ::std::size_t size() const override {
    PYBIND11_OVERLOAD_PURE(::std::size_t, NearestNeighbors, size);
  }
};

class PySampler : public Sampler {
 public:
  using Sampler::Sampler;
  virtual ::rl::math::Vector generate() override {
    PYBIND11_OVERLOAD_PURE(::rl::math::Vector, Sampler, generate);
  }
};

class PyViewer : public Viewer {
 public:
  using Viewer::Viewer;
  virtual void drawConfiguration(const ::rl::math::Vector& q) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawConfiguration, q);
  }

  virtual void drawConfigurationEdge(const ::rl::math::Vector& q0,
                                     const ::rl::math::Vector& q1,
                                     const bool& free = true) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawConfigurationEdge, q0, q1, free);
  }

  virtual void drawConfigurationPath(const VectorList& path) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawConfigurationPath, path);
  }

  virtual void drawConfigurationVertex(const ::rl::math::Vector& q,
                                       const bool& free = true) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawConfigurationVertex, q, free);
  }

  virtual void drawLine(const ::rl::math::Vector& xyz0,
                        const ::rl::math::Vector& xyz1) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawLine, xyz0, xyz1);
  }

  virtual void drawPoint(const ::rl::math::Vector& xyz) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawPoint, xyz);
  }

  virtual void drawSphere(const ::rl::math::Vector& center,
                          const ::rl::math::Real& radius) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawSphere, center, radius);
  }

  virtual void drawWork(const ::rl::math::Transform& t) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawWork, t);
  }

  virtual void drawWorkEdge(const ::rl::math::Vector& q0,
                            const ::rl::math::Vector& q1) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawWorkEdge, q0, q1);
  }

  virtual void drawWorkPath(const VectorList& path) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawWorkPath, path);
  }

  virtual void drawWorkVertex(const ::rl::math::Vector& q) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, drawWorkVertex, q);
  }

  virtual void reset() override { PYBIND11_OVERLOAD_PURE(void, Viewer, reset); }

  virtual void resetEdges() override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, resetEdges);
  }

  virtual void resetLines() override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, resetLines);
  }

  virtual void resetPoints() override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, resetPoints);
  }

  virtual void resetSpheres() override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, resetSpheres);
  }

  virtual void resetVertices() override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, resetVertices);
  }

  virtual void showMessage(const ::std::string& message) override {
    PYBIND11_OVERLOAD_PURE(void, Viewer, showMessage, message);
  }
};

class PyVerifier : public Verifier {
 public:
  using Verifier::Verifier;
  virtual bool isColliding(const ::rl::math::Vector& u,
                           const ::rl::math::Vector& v,
                           const ::rl::math::Real& d) override {
    PYBIND11_OVERLOAD_PURE(bool, Verifier, isColliding, u, v, d);
  }
};

class PyPlanner : public Planner {
 public:
  using Planner::Planner;
  virtual ::std::string getName() const override {
    PYBIND11_OVERLOAD_PURE(::std::string, Planner, getName);
  }
  virtual VectorList getPath() override {
    PYBIND11_OVERLOAD_PURE(VectorList, Planner, getPath);
  }
  virtual void reset() override {
    PYBIND11_OVERLOAD_PURE(void, Planner, reset);
  }
  virtual bool solve() override {
    PYBIND11_OVERLOAD_PURE(bool, Planner, solve);
  }
};

class PyPrm : public Prm {
 public:
  PyPrm() {
    start = &_start;
    goal = &_goal;
  }
  ::rl::math::Vector _start, _goal;
};

class PyRrt : public Rrt {
 public:
  PyRrt() {
    start = &_start;
    goal = &_goal;
  }
  ::rl::math::Vector _start, _goal;
};

}  // namespace plan
}  // namespace rl

namespace py = pybind11;
using namespace rl;

PYBIND11_MODULE(pyrol, m) {
  m.doc() =
      "self-contained C++ library for rigid body kinematics and dynamics, "
      "motion planning, and control";

  {
    py::module math = m.def_submodule(
        "math", "General mathematical data structures and algorithms");

    py::class_<math::AngleAxis>(math, "AngleAxis")
        .def(py::init<>())
        .def(py::init<const math::Real&, const math::Vector3&>())
        .def(py::init<const math::Quaternion&>())
        .def(py::init<const math::Matrix33>())
        .def(py::self * py::self)
        .def(py::self * math::Quaternion());

    py::class_<math::Quaternion>(math, "Quaternion")
        .def(py::init<>())
        .def(py::init<const math::Real&, const math::Real&, const math::Real&,
                      const math::Real&>())
        .def(py::init<const math::Real*>())
        .def(py::init<const math::AngleAxis&>())
        .def(py::init<const math::Matrix33&>())
        .def(py::init<const math::Vector4&>())
        .def_static("fromEulerAngles",
                    [](const math::Vector3& euler_angles) {
                      return math::Quaternion(
                          Eigen::AngleAxisd(euler_angles[2],
                                            rl::math::Vector3::UnitZ()) *
                          Eigen::AngleAxisd(euler_angles[1],
                                            rl::math::Vector3::UnitY()) *
                          Eigen::AngleAxisd(euler_angles[0],
                                            rl::math::Vector3::UnitX()));
                    })
        .def(py::self * py::self)
        .def(py::self * math::AngleAxis())
        .def("__repr__", [](const math::Quaternion& quat) {
          std::stringstream repr_stream;
          repr_stream << "( "
                      << quat.vec().format(
                             Eigen::IOFormat(5, 0, " ", " ", "", "", "[", "]"))
                      << " " << quat.w() << " )";
          return repr_stream.str();
        });

    py::class_<math::Transform>(math, "Transform")
        .def(py::init<>())
        .def_static("Identity", &math::Transform::Identity)
        .def("translation",
             [](math::Transform const& transform) {
               return math::Vector3(transform.translation());
             })
        .def("rotation",
             [](math::Transform const& transform) {
               return math::Matrix33(transform.rotation());
             })
        .def("eulerAngles",
             [](math::Transform const& transform) {
               return math::Vector3(
                   transform.rotation().eulerAngles(2, 1, 0).reverse());
             })
        .def("quaternion",
             [](math::Transform const& transform) {
               math::Quaternion quat(transform.rotation());
               return math::Vector4(quat.x(), quat.y(), quat.z(), quat.w());
             })
        .def("rotate",
             [](math::Transform& transform, const math::Quaternion& quat) {
               return transform.rotate(quat);
             })
        .def("setIdentity", &math::Transform::setIdentity)
        .def("translate",
             [](math::Transform& transform, const math::Vector3& v) {
               return transform.translate(v);
             });
  }
  {
    py::module mdl = m.def_submodule(
        "mdl",
        "Kinematic and dynamic spatial vector calculations, including several "
        "joint types and branched kinematics");
    py::class_<mdl::XmlFactory>(mdl, "XmlFactory")
        .def(py::init<>())
        .def("load", &mdl::XmlFactory::load);

    py::class_<mdl::Model>(mdl, "Model")
        .def(py::init<>())
        .def_property("name", &mdl::Model::getName, &mdl::Model::setName)
        .def_property_readonly("bodies", &mdl::Model::getBodies)
        .def_property_readonly("frames", &mdl::Model::getFrames)
        .def_property("gammaPosition", &mdl::Model::getGammaPosition,
                      &mdl::Model::setGammaPosition)
        .def_property("gammaVelocity", &mdl::Model::getGammaVelocity,
                      &mdl::Model::setGammaVelocity)
        .def_property("home", &mdl::Model::getHomePosition,
                      &mdl::Model::setHomePosition)
        .def_property_readonly("invGammaPosition",
                               &mdl::Model::getGammaPositionInverse)
        .def_property_readonly("invGammaVelocity",
                               &mdl::Model::getGammaVelocityInverse)
        .def_property_readonly("joints", &mdl::Model::getJoints)
        .def_property("manufacturer", &mdl::Model::getManufacturer,
                      &mdl::Model::setManufacturer)
        .def_property_readonly("transforms", &mdl::Model::getTransforms)
        .def("add", py::overload_cast<mdl::Compound*, const mdl::Frame*,
                                      const mdl::Frame*>(&mdl::Model::add))
        .def("add", py::overload_cast<mdl::Frame*>(&mdl::Model::add))
        .def("add", py::overload_cast<mdl::Transform*, const mdl::Frame*,
                                      const mdl::Frame*>(&mdl::Model::add))
        .def("areColliding", &mdl::Model::areColliding)
        .def("generatePositionGaussian",
             py::overload_cast<const ::rl::math::Vector&,
                               const ::rl::math::Vector&>(
                 &mdl::Model::generatePositionGaussian))
        .def("generatePositionGaussian",
             py::overload_cast<const ::rl::math::Vector&,
                               const ::rl::math::Vector&,
                               const ::rl::math::Vector&>(
                 &mdl::Model::generatePositionGaussian, py::const_))
        .def("generatePositionUniform",
             py::overload_cast<>(&mdl::Model::generatePositionUniform))
        .def("generatePositionUniform",
             py::overload_cast<const ::rl::math::Vector&>(
                 &mdl::Model::generatePositionUniform, py::const_))
        .def("generatePositionUniform",
             py::overload_cast<const ::rl::math::Vector&,
                               const ::rl::math::Vector&>(
                 &mdl::Model::generatePositionUniform))
        .def("generatePositionUniform",
             py::overload_cast<const ::rl::math::Vector&,
                               const ::rl::math::Vector&,
                               const ::rl::math::Vector&>(
                 &mdl::Model::generatePositionUniform, py::const_))
        .def("getAcceleration", &mdl::Model::getAcceleration)
        .def("getAccelerationUnits", &mdl::Model::getAccelerationUnits)
        .def("getBodies", &mdl::Model::getBodies)
        .def("getBody", &mdl::Model::getBody)
        .def("getBodyFrame", &mdl::Model::getBodyFrame)
        .def("getDof", &mdl::Model::getDof)
        .def("getDofPosition", &mdl::Model::getDofPosition)
        .def("getFrame", &mdl::Model::getFrame)
        .def("getFrames", &mdl::Model::getFrames)
        .def("getGammaPosition", &mdl::Model::getGammaPosition)
        .def("getGammaVelocity", &mdl::Model::getGammaVelocity)
        .def("getGammaPositionInverse", &mdl::Model::getGammaPositionInverse)
        .def("getGammaVelocityInverse", &mdl::Model::getGammaVelocityInverse)
        .def("getHomePosition", &mdl::Model::getHomePosition)
        .def("getJoint", &mdl::Model::getJoint)
        .def("getJoints", &mdl::Model::getJoints)
        .def("getOperationalAcceleration",
             &mdl::Model::getOperationalAcceleration)
        .def("getOperationalDof", &mdl::Model::getOperationalDof)
        .def("getOperationalForce", &mdl::Model::getOperationalForce)
        .def("getOperationalPosition", &mdl::Model::getOperationalPosition)
        .def("getOperationalVelocity", &mdl::Model::getOperationalVelocity)
        .def("getManufacturer", &mdl::Model::getManufacturer)
        .def("getMaximum", &mdl::Model::getMaximum)
        .def("getMinimum", &mdl::Model::getMinimum)
        .def("getName", &mdl::Model::getName)
        .def("getPosition", &mdl::Model::getPosition)
        .def("getPositionUnits", &mdl::Model::getPositionUnits)
        .def("getTransform", &mdl::Model::getTransform)
        .def("getTransforms", &mdl::Model::getTransforms)
        .def("getSpeed", &mdl::Model::getSpeed)
        .def("getSpeedUnits", &mdl::Model::getSpeedUnits)
        .def("getTorque", &mdl::Model::getTorque)
        .def("getTorqueUnits", &mdl::Model::getTorqueUnits)
        .def("getVelocity", &mdl::Model::getVelocity)
        .def("getVelocityUnits", &mdl::Model::getVelocityUnits)
        .def("getWorld", &mdl::Model::getWorld)
        .def("getWorldGravity", &mdl::Model::getWorldGravity)
        .def("getWraparounds", &mdl::Model::getWraparounds)
        .def("isColliding", &mdl::Model::isColliding)
        .def("replace", py::overload_cast<mdl::Transform*, mdl::Compound*>(
                            &mdl::Model::replace))
        .def("remove", py::overload_cast<mdl::Compound*>(&mdl::Model::remove))
        .def("remove", py::overload_cast<mdl::Frame*>(&mdl::Model::remove))
        .def("remove", py::overload_cast<mdl::Transform*>(&mdl::Model::remove))
        .def("seed", &mdl::Model::seed)
        .def("setAcceleration", &mdl::Model::setAcceleration)
        .def("setGammaPosition", &mdl::Model::setGammaPosition)
        .def("setGammaVelocity", &mdl::Model::setGammaVelocity)
        .def("setHomePosition", &mdl::Model::setHomePosition)
        .def("setManufacturer", &mdl::Model::setManufacturer)
        .def("setName", &mdl::Model::setName)
        .def("setOperationalVelocity", &mdl::Model::setOperationalVelocity)
        .def("setPosition", &mdl::Model::setPosition)
        .def("setTorque", &mdl::Model::setTorque)
        .def("setVelocity", &mdl::Model::setVelocity)
        .def("setWorldGravity", &mdl::Model::setWorldGravity)
        .def("tool", py::overload_cast<const ::std::size_t&>(&mdl::Model::tool))
        .def("tool", py::overload_cast<const ::std::size_t&>(&mdl::Model::tool,
                                                             py::const_))
        .def("world", py::overload_cast<>(&mdl::Model::world))
        .def("world", py::overload_cast<>(&mdl::Model::world, py::const_));

    py::class_<mdl::Kinematic, mdl::Model>(mdl, "Kinematic")
        .def(py::init<>())
        .def_property_readonly("invJ", &mdl::Kinematic::getJacobianInverse)
        .def_property_readonly("J", &mdl::Kinematic::getJacobian)
        .def_property_readonly("Jdqd", &mdl::Kinematic::getJacobianDerivative)
        .def("calculateJacobian",
             py::overload_cast<const bool&>(&mdl::Kinematic::calculateJacobian))
        .def("calculateJacobian",
             py::overload_cast<::rl::math::Matrix&, const bool&>(
                 &mdl::Kinematic::calculateJacobian))
        .def("calculateJacobianDerivative",
             py::overload_cast<const bool&>(
                 &mdl::Kinematic::calculateJacobianDerivative))
        .def("calculateJacobianDerivative",
             py::overload_cast<::rl::math::Vector&, const bool&>(
                 &mdl::Kinematic::calculateJacobianDerivative))
        .def("calculateJacobianInverse",
             py::overload_cast<const ::rl::math::Real&, const bool&>(
                 &mdl::Kinematic::calculateJacobianInverse))
        .def("calculateJacobianInverse",
             py::overload_cast<const ::rl::math::Matrix&, ::rl::math::Matrix&,
                               const ::rl::math::Real&, const bool&>(
                 &mdl::Kinematic::calculateJacobianInverse, py::const_))
        .def("calculateManipulabilityMeasure",
             py::overload_cast<>(
                 &mdl::Kinematic::calculateManipulabilityMeasure, py::const_))
        .def("calculateManipulabilityMeasure",
             py::overload_cast<const ::rl::math::Matrix&>(
                 &mdl::Kinematic::calculateManipulabilityMeasure, py::const_))
        .def("forwardAcceleration", &mdl::Kinematic::forwardAcceleration)
        .def("forwardPosition", &mdl::Kinematic::forwardPosition)
        .def("forwardVelocity", &mdl::Kinematic::forwardVelocity)
        .def("getJacobian", &mdl::Kinematic::getJacobian)
        .def("getJacobianDerivative", &mdl::Kinematic::getJacobianDerivative)
        .def("getJacobianInverse", &mdl::Kinematic::getJacobianInverse)
        .def("isSingular",
             py::overload_cast<>(&mdl::Kinematic::isSingular, py::const_))
        .def("isSingular", py::overload_cast<const ::rl::math::Matrix&>(
                               &mdl::Kinematic::isSingular, py::const_));

    py::class_<mdl::InverseKinematics, mdl::PyInverseKinematics>(
        mdl, "InverseKinematics")
        .def(py::init<mdl::Kinematic*>())
        .def_property_readonly("goals", &mdl::InverseKinematics::getGoals)
        .def("addGoal", py::overload_cast<mdl::InverseKinematics::Goal const&>(
                            &mdl::InverseKinematics::addGoal))
        .def("addGoal", py::overload_cast<const ::rl::math::Transform&,
                                          const ::std::size_t&>(
                            &mdl::InverseKinematics::addGoal))
        .def("clearGoals", &mdl::InverseKinematics::clearGoals)
        .def("getGoals", &mdl::InverseKinematics::getGoals)
        .def("solve", &mdl::InverseKinematics::solve);

    py::class_<mdl::IterativeInverseKinematics,
               mdl::PyIterativeInverseKinematics, mdl::InverseKinematics>(
        mdl, "IterativeInverseKinematics")
        .def(py::init<mdl::Kinematic*>())
        .def_property("duration", &mdl::IterativeInverseKinematics::getDuration,
                      &mdl::IterativeInverseKinematics::setDuration)
        .def_property("epsilon", &mdl::IterativeInverseKinematics::getEpsilon,
                      &mdl::IterativeInverseKinematics::setEpsilon)
        .def_property("iterations",
                      &mdl::IterativeInverseKinematics::getIterations,
                      &mdl::IterativeInverseKinematics::setIterations);

    py::class_<mdl::JacobianInverseKinematics, mdl::IterativeInverseKinematics>(
        mdl, "JacobianInverseKinematics")
        .def(py::init<mdl::Kinematic*>())
        .def_property("delta", &mdl::JacobianInverseKinematics::getDelta,
                      &mdl::JacobianInverseKinematics::setDelta)
        .def_property("method", &mdl::JacobianInverseKinematics::getMethod,
                      &mdl::JacobianInverseKinematics::setMethod)
        .def("seed", &mdl::JacobianInverseKinematics::seed);

    py::class_<mdl::NloptInverseKinematics, mdl::IterativeInverseKinematics>(
        mdl, "NloptInverseKinematics")
        .def(py::init<mdl::Kinematic*>())
        .def_property("lb", &mdl::NloptInverseKinematics::getLowerBound,
                      &mdl::NloptInverseKinematics::setLowerBound)
        .def_property("up", &mdl::NloptInverseKinematics::getUpperBound,
                      &mdl::NloptInverseKinematics::setUpperBound)
        .def_property("iteration", &mdl::NloptInverseKinematics::getIterations,
                      &mdl::NloptInverseKinematics::setIterations)
        .def("getFunctionToleranceAbsolute",
             &mdl::NloptInverseKinematics::getFunctionToleranceAbsolute)
        .def("getFunctionToleranceRelative",
             &mdl::NloptInverseKinematics::getFunctionToleranceRelative)
        .def("getOptimizationToleranceAbsolute",
             &mdl::NloptInverseKinematics::getOptimizationToleranceAbsolute)
        .def("getOptimizationToleranceRelative",
             &mdl::NloptInverseKinematics::getOptimizationToleranceRelative)
        .def("seed", &mdl::NloptInverseKinematics::seed)
        .def("setFunctionToleranceAbsolute",
             &mdl::NloptInverseKinematics::setFunctionToleranceAbsolute)
        .def("setFunctionToleranceRelative",
             &mdl::NloptInverseKinematics::setFunctionToleranceRelative)
        .def(
            "setOptimizationToleranceAbsolute",
            py::overload_cast<const ::rl::math::Real&>(
                &mdl::NloptInverseKinematics::setOptimizationToleranceAbsolute))
        .def(
            "setOptimizationToleranceAbsolute",
            py::overload_cast<const ::rl::math::Vector&>(
                &mdl::NloptInverseKinematics::setOptimizationToleranceAbsolute))
        .def("setOptimizationToleranceRelative",
             &mdl::NloptInverseKinematics::setOptimizationToleranceRelative);
  }

  {
    py::module sg = m.def_submodule(
        "sg",
        "Scene graph abstraction for collision checking, distance "
        "queries, raycasts, and visualization");
    py::class_<sg::XmlFactory>(sg, "XmlFactory")
        .def(py::init<>())
        .def("load", py::overload_cast<const ::std::string&, sg::Scene*>(
                         &sg::XmlFactory::load))
        .def("load",
             py::overload_cast<const ::std::string&, sg::Scene*, const bool&,
                               const bool&>(&sg::XmlFactory::load));

    py::class_<sg::Shape, sg::PyShape>(sg, "Shape")
        .def(py::init<::SoVRMLShape*, sg::Body*>())
        .def_property("name", &sg::Shape::getName, &sg::Shape::setName)
        .def("getBody", &sg::Shape::getBody, py::return_value_policy::reference)
        .def("getTransform", &sg::Shape::getTransform)
        .def("setTransform", &sg::Shape::setTransform);

    py::class_<sg::Body, sg::PyBody>(sg, "Body")
        .def(py::init<sg::Model*>())
        .def_readwrite("center", &sg::Body::center)
        .def_readwrite("max", &sg::Body::max)
        .def_readwrite("min", &sg::Body::min)
        .def_property("name", &sg::Body::getName, &sg::Body::setName)
        .def("add", &sg::Body::add)
        .def("remove", &sg::Body::remove)
        .def("getModel", &sg::Body::getModel,
             py::return_value_policy::reference)
        .def("getShape", &sg::Body::getShape,
             py::return_value_policy::reference)
        .def("getNumShapes", &sg::Body::getNumShapes);

    py::class_<sg::Model, sg::PyModel>(sg, "Model")
        .def(py::init<sg::Scene*>())
        .def_property("name", &sg::Model::getName, &sg::Model::setName)
        .def("create", &sg::Model::create)
        .def("add", &sg::Model::add)
        .def("remove", &sg::Model::remove)
        .def("getBody", &sg::Model::getBody, py::return_value_policy::reference)
        .def("getNumBodies", &sg::Model::getNumBodies);

    py::class_<sg::Scene, sg::PyScene>(sg, "Scene")
        .def(py::init<>())
        .def_property("name", &sg::Scene::getName, &sg::Scene::setName)
        .def("create", &sg::Scene::create)
        .def("add", &sg::Scene::add)
        .def("remove", &sg::Scene::remove)
        .def("getModel", &sg::Scene::getModel,
             py::return_value_policy::reference)
        .def("getNumModels", &sg::Scene::getNumModels);
    {
      py::module bullet = sg.def_submodule("bullet", "Bullet Physics Library");
      py::class_<sg::bullet::Scene, sg::Scene>(bullet, "Scene",
                                               py::multiple_inheritance())
          .def(py::init<>());
    }
  }
  {
    py::module plan = m.def_submodule("plan", "Robot path planning algorithms");

    py::class_<plan::Sampler, plan::PySampler>(plan, "Sampler")
        .def(py::init<>())
        .def_readwrite("model", &plan::Sampler::model)
        .def("generate", &plan::Sampler::generate)
        .def("generateCollisionFree", &plan::Sampler::generateCollisionFree);

    py::class_<plan::UniformSampler, plan::Sampler>(plan, "UniformSampler")
        .def(py::init<>())
        .def("seed", &plan::UniformSampler::seed);

    py::class_<plan::GaussianSampler, plan::UniformSampler>(plan,
                                                            "GaussianSampler")
        .def(py::init<>())
        .def_readwrite("sigma", &plan::GaussianSampler::sigma);

    py::class_<plan::BridgeSampler, plan::GaussianSampler>(plan,
                                                           "BridgeSampler")
        .def(py::init<>())
        .def_readwrite("ratio", &plan::BridgeSampler::ratio);

    py::class_<plan::Verifier, plan::PyVerifier>(plan, "Verifier")
        .def(py::init<>())
        .def_readwrite("delta", &plan::Verifier::delta)
        .def_readwrite("model", &plan::Verifier::model)
        .def("getSteps", &plan::Verifier::getSteps)
        .def("isColliding", &plan::Verifier::isColliding);

    py::class_<plan::SequentialVerifier, plan::Verifier>(plan,
                                                         "SequentialVerifier")
        .def(py::init<>());

    py::class_<plan::RecursiveVerifier, plan::Verifier>(plan,
                                                        "RecursiveVerifier")
        .def(py::init<>());

    py::class_<plan::NearestNeighbors, plan::PyNearestNeighbors>(
        plan, "NearestNeighbors")
        .def(py::init<const bool&>())
        .def("clear", &plan::NearestNeighbors::clear)
        .def("empty", &plan::NearestNeighbors::empty)
        .def("isTransformedDistance",
             &plan::NearestNeighbors::isTransformedDistance)
        .def("nearest", &plan::NearestNeighbors::nearest)
        .def("push", &plan::NearestNeighbors::push)
        .def("radius", &plan::NearestNeighbors::radius)
        .def("size", &plan::NearestNeighbors::size);

    py::class_<plan::LinearNearestNeighbors, plan::NearestNeighbors>(
        plan, "LinearNearestNeighbors")
        .def(py::init<plan::Model*>());

    py::class_<plan::KdtreeNearestNeighbors, plan::NearestNeighbors>(
        plan, "KdtreeNearestNeighbors")
        .def(py::init<plan::Model*>())
        .def("getChecks", &plan::KdtreeNearestNeighbors::getChecks)
        .def("getSamples", &plan::KdtreeNearestNeighbors::getSamples)
        .def("setChecks", &plan::KdtreeNearestNeighbors::setChecks)
        .def("setSamples", &plan::KdtreeNearestNeighbors::setSamples);

    py::class_<plan::KdtreeBoundingBoxNearestNeighbors, plan::NearestNeighbors>(
        plan, "KdtreeBoundingBoxNearestNeighbors")
        .def(py::init<plan::Model*>())
        .def("getChecks", &plan::KdtreeBoundingBoxNearestNeighbors::getChecks)
        .def("getNodeDataMax",
             &plan::KdtreeBoundingBoxNearestNeighbors::getNodeDataMax)
        .def("setChecks", &plan::KdtreeBoundingBoxNearestNeighbors::setChecks)
        .def("setNodeDataMax",
             &plan::KdtreeBoundingBoxNearestNeighbors::setNodeDataMax);

    py::class_<plan::GnatNearestNeighbors, plan::NearestNeighbors>(
        plan, "GnatNearestNeighbors")
        .def(py::init<plan::Model*>())
        .def("getChecks", &plan::GnatNearestNeighbors::getChecks)
        .def("getNodeDataMax", &plan::GnatNearestNeighbors::getNodeDataMax)
        .def("getNodeDegree", &plan::GnatNearestNeighbors::getNodeDegree)
        .def("getNodeDegreeMax", &plan::GnatNearestNeighbors::getNodeDegreeMax)
        .def("getNodeDegreeMin", &plan::GnatNearestNeighbors::getNodeDegreeMin)
        .def("seed", &plan::GnatNearestNeighbors::seed)
        .def("setChecks", &plan::GnatNearestNeighbors::setChecks)
        .def("setNodeDataMax", &plan::GnatNearestNeighbors::setNodeDataMax)
        .def("setNodeDegree", &plan::GnatNearestNeighbors::setNodeDegree)
        .def("setNodeDegreeMax", &plan::GnatNearestNeighbors::setNodeDegreeMax)
        .def("setNodeDegreeMin", &plan::GnatNearestNeighbors::setNodeDegreeMin);

    py::class_<plan::Model>(plan, "Model")
        .def(py::init<>())
        .def_readwrite("kin", &plan::Model::kin)
        .def_readwrite("mdl", &plan::Model::mdl)
        .def_readwrite("model", &plan::Model::model)
        .def_readwrite("scene", &plan::Model::scene)
        .def("areColliding", &plan::Model::areColliding)
        .def("clamp", &plan::Model::clamp)
        .def("distance", &plan::Model::distance)
        .def("forwardForce", &plan::Model::forwardForce)
        .def("forwardPosition", &plan::Model::forwardPosition)
        .def("generatePositionGaussian", &plan::Model::generatePositionGaussian)
        .def("generatePositionUniform", &plan::Model::generatePositionUniform)
        .def("getBody", &plan::Model::getBody)
        .def("getBodies", &plan::Model::getBodies)
        .def("getCenter", &plan::Model::getCenter)
        .def("getDof", &plan::Model::getDof)
        .def("getDofPosition", &plan::Model::getDofPosition)
        .def("getFrame", &plan::Model::getFrame)
        .def("getJacobian", &plan::Model::getJacobian)
        .def("getManipulabilityMeasure", &plan::Model::getManipulabilityMeasure)
        .def("getManufacturer", &plan::Model::getManufacturer)
        .def("getMaximum", &plan::Model::getMaximum)
        .def("getMinimum", &plan::Model::getMinimum)
        .def("getName", &plan::Model::getName)
        .def("getOperationalDof", &plan::Model::getOperationalDof)
        .def("getPositionUnits", &plan::Model::getPositionUnits)
        .def("getWraparounds", &plan::Model::getWraparounds)
        .def("inverseForce", &plan::Model::inverseForce)
        .def("inverseOfTransformedDistance",
             &plan::Model::inverseOfTransformedDistance)
        .def("inverseVelocity", &plan::Model::inverseVelocity)
        .def("interpolate", &plan::Model::interpolate)
        .def("isColliding", &plan::Model::isColliding)
        .def("isSingular", &plan::Model::isSingular)
        .def("isValid", &plan::Model::isValid)
        .def("reset", &plan::Model::reset)
        .def("setPosition", &plan::Model::setPosition)
        .def("step", &plan::Model::step)
        .def("updateFrames", &plan::Model::updateFrames)
        .def("updateJacobian", &plan::Model::updateJacobian)
        .def("updateJacobianInverse", &plan::Model::updateJacobianInverse);

    py::class_<plan::SimpleModel, plan::Model>(plan, "SimpleModel")
        .def(py::init<>())
        .def("getCollidingBody", &plan::SimpleModel::getCollidingBody)
        .def("getFreeQueries", &plan::SimpleModel::getFreeQueries)
        .def("getTotalQueries", &plan::SimpleModel::getTotalQueries)
        .def("isColliding",
             py::overload_cast<>(&plan::SimpleModel::isColliding))
        .def("isColliding", py::overload_cast<const ::rl::math::Vector&>(
                                &plan::SimpleModel::isColliding))
        .def("reset", &plan::SimpleModel::reset);

    py::class_<plan::Planner, plan::PyPlanner>(plan, "Planner")
        .def(py::init<>())
        .def_readwrite("duration", &plan::Planner::duration)
        .def_readwrite("model", &plan::Planner::model)
        .def_readwrite("viewer", &plan::Planner::viewer)
        .def("getName", &plan::Planner::getName)
        .def("getPath", &plan::Planner::getPath)
        .def("reset", &plan::Planner::reset)
        .def("solve", &plan::Planner::solve)
        .def("verify", &plan::Planner::verify);

    py::class_<plan::PyRrt, plan::Planner>(plan, "Rrt")
        .def(py::init<>())
        .def_readwrite("start", &plan::PyRrt::_start)
        .def_readwrite("goal", &plan::PyRrt::_goal)
        .def_readwrite("delta", &plan::Rrt::delta)
        .def_readwrite("epsilon", &plan::Rrt::epsilon)
        .def_readwrite("sampler", &plan::Rrt::sampler)
        .def("getNearestNeighbors", &plan::Rrt::getNearestNeighbors)
        .def("getNumEdges", &plan::Rrt::getNumEdges)
        .def("getNumVertices", &plan::Rrt::getNumVertices)
        .def("setNearestNeighbors", &plan::Rrt::setNearestNeighbors);

    py::class_<plan::PyPrm, plan::Planner>(plan, "Prm")
        .def(py::init<>())
        .def_readwrite("start", &plan::PyPrm::_start)
        .def_readwrite("goal", &plan::PyPrm::_goal)
        .def_readwrite("astar", &plan::Prm::astar)
        .def_readwrite("degree", &plan::Prm::degree)
        .def_readwrite("k", &plan::Prm::k)
        .def_readwrite("radius", &plan::Prm::radius)
        .def_readwrite("sampler", &plan::Prm::sampler)
        .def_readwrite("verifier", &plan::Prm::verifier)
        .def("construct", &plan::Prm::construct)
        .def("getNearestNeighbors", &plan::Prm::getNearestNeighbors)
        .def("getNumEdges", &plan::Prm::getNumEdges)
        .def("getNumVertices", &plan::Prm::getNumVertices)
        .def("setNearestNeighbors", &plan::Prm::setNearestNeighbors);
  }
}
