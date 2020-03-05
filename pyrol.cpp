#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <rl/mdl/Kinematic.h>
#include <rl/mdl/Model.h>
#include <rl/mdl/XmlFactory.h>

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
#include <utility>

namespace rl {
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

class VectorWrapper {
 public:
  VectorWrapper(::rl::math::Vector const& v) : q(v) {}
  ::rl::math::Vector q;
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
    py::class_<math::Transform>(math, "Transform")
        .def(py::init<>())
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
        .def("quaternion", [](math::Transform const& transform) {
          math::Quaternion quat(transform.rotation());
          return math::Vector4(quat.x(), quat.y(), quat.z(), quat.w());
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
        .def("setPosition", &mdl::Model::setPosition)
        .def("areColliding", &mdl::Model::areColliding)
        .def("isColliding", &mdl::Model::isColliding)
        .def("getOperationalPosition", &mdl::Model::getOperationalPosition);

    py::class_<mdl::Kinematic, mdl::Model>(mdl, "Kinematic")
        .def(py::init<>())
        .def("forwardPosition", &mdl::Kinematic::forwardPosition);
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
        .def_readonly("start", &plan::Planner::start)
        .def("setStart",
             [](plan::Planner& planner, plan::VectorWrapper& wrapped) {
               planner.start = &wrapped.q;
             })
        .def_readonly("goal", &plan::Planner::goal)
        .def("setGoal",
             [](plan::Planner& planner, plan::VectorWrapper& wrapped) {
               planner.goal = &wrapped.q;
             })
        .def_readwrite("model", &plan::Planner::model)
        .def_readwrite("viewer", &plan::Planner::viewer)
        .def("getName", &plan::Planner::getName)
        .def("getPath", &plan::Planner::getPath)
        .def("reset", &plan::Planner::reset)
        .def("solve", &plan::Planner::solve)
        .def("verify", &plan::Planner::verify);

    py::class_<plan::Rrt, plan::Planner>(plan, "Rrt")
        .def(py::init<>())
        .def_readwrite("delta", &plan::Rrt::delta)
        .def_readwrite("epsilon", &plan::Rrt::epsilon)
        .def_readwrite("sampler", &plan::Rrt::sampler)
        .def("getNearestNeighbors", &plan::Rrt::getNearestNeighbors)
        .def("getNumEdges", &plan::Rrt::getNumEdges)
        .def("getNumVertices", &plan::Rrt::getNumVertices)
        .def("setNearestNeighbors", &plan::Rrt::setNearestNeighbors);

    py::class_<plan::Prm, plan::Planner>(plan, "Prm")
        .def(py::init<>())
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

    py::class_<plan::VectorWrapper>(plan, "VectorWrapper")
        .def(py::init<const math::Vector&>());
  }
}
