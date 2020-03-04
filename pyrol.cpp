#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <rl/mdl/Kinematic.h>
#include <rl/mdl/Model.h>
#include <rl/mdl/XmlFactory.h>
#include <rl/sg/Model.h>
#include <rl/sg/XmlFactory.h>
#include <rl/sg/bullet/Scene.h>
#include <utility>

namespace rl {
namespace sg {
class PyScene : public Scene {
 public:
  using Scene::Scene;
  virtual Model* create() override {
    PYBIND11_OVERLOAD_PURE(Model*, Scene, create);
  }
};
}  // namespace sg
}  // namespace rl

namespace py = pybind11;
namespace sg = rl::sg;
namespace mdl = rl::mdl;
namespace math = rl::math;

PYBIND11_MODULE(pyrol, m) {
  m.doc() =
      "self-contained C++ library for rigid body kinematics and dynamics, "
      "motion planning, and control";
  py::module mdl = m.def_submodule(
      "mdl",
      "Kinematic and dynamic spatial vector calculations, including several "
      "joint types and branched kinematics");

  {
    py::class_<mdl::XmlFactory> xml_factory(mdl, "XmlFactory");
    xml_factory.def(py::init<>()).def("load", &mdl::XmlFactory::load);

    py::class_<mdl::Model> model(mdl, "Model");
    model.def(py::init<>())
        .def_property("name", &mdl::Model::getName, &mdl::Model::setName)
        .def("setPosition", &mdl::Model::setPosition)
        .def("areColliding", &mdl::Model::areColliding)
        .def("isColliding", &mdl::Model::isColliding)
        .def("getOperationalPosition",
             [](mdl::Model const& model, const ::std::size_t& i) {
               math::Transform t = model.getOperationalPosition(i);
               return std::make_pair(
                   math::Vector3(t.translation()),
                   math::Vector3(t.rotation().eulerAngles(2, 1, 0).reverse()));
             });

    py::class_<mdl::Kinematic> kinematic(mdl, "Kinematic", model);
    kinematic.def(py::init<>())
        .def("forwardPosition", &mdl::Kinematic::forwardPosition);
  }

  {
    py::module sg = m.def_submodule(
        "sg",
        "cene graph abstraction for collision checking, distance "
        "queries, raycasts, and visualization");
    py::class_<sg::XmlFactory> xml_factory(sg, "XmlFactory");
    xml_factory.def(py::init<>())
        .def("load", py::overload_cast<const ::std::string&, sg::Scene*>(
                         &sg::XmlFactory::load))
        .def("load",
             py::overload_cast<const ::std::string&, sg::Scene*, const bool&,
                               const bool&>(&sg::XmlFactory::load));

    py::class_<sg::Scene, sg::PyScene> scene(sg, "Scene");
    scene.def(py::init<>())
        .def("create", &sg::Scene::create)
        .def_property("name", &sg::Scene::getName, &sg::Scene::setName)
        .def("add", &sg::Scene::add)
        .def("remove", &sg::Scene::remove)
        .def("getModel", &sg::Scene::getModel)
        .def("getNumModels", &sg::Scene::getNumModels);
    {
      py::module bullet = sg.def_submodule("bullet", "Bullet Physics Library");
      py::class_<sg::bullet::Scene, sg::Scene> bullet_scene(
          bullet, "Scene", py::multiple_inheritance());
      bullet_scene.def(py::init<>());
    }
  }
}
