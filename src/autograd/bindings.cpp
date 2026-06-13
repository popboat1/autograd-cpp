#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Value.h"
#include "nn/Linear.h"
#include "nn/MLP.h"
#include "optim/SGD.h"
#include "nn/Loss.h"

namespace py = pybind11;

PYBIND11_MODULE(autograd_cpp, m) {
    m.doc() = "C++ Autograd Engine Python Bindings";

    py::class_<Value, std::shared_ptr<Value>>(m, "Value")
        .def(py::init<double, bool>(), py::arg("val"), py::arg("requires_grad") = true)
        .def_readwrite("data", &Value::data)
        .def_readwrite("grad", &Value::grad)
        .def_readwrite("op", &Value::op)
        .def_property_readonly("requires_grad", [](const ValuePtr& self) { return self->requires_grad; })
        
        .def("backward", &Value::backward)
        .def("pow", &Value::pow)
        .def("tanh", &Value::tanh)
        .def("exp", &Value::exp)
        .def("relu", &Value::relu)
        .def("print", &Value::print)

        .def("__add__", [](const ValuePtr& lhs, const ValuePtr& rhs) { return lhs + rhs; })
        .def("__mul__", [](const ValuePtr& lhs, const ValuePtr& rhs) { return lhs * rhs; })
        .def("__sub__", [](const ValuePtr& lhs, const ValuePtr& rhs) { return lhs - rhs; })
        .def("__truediv__", [](const ValuePtr& lhs, const ValuePtr& rhs) { return lhs / rhs; })
        .def("__pow__", [](const ValuePtr& self, double exponent) { return self->pow(exponent); });
    
    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int, int>(), py::arg("fan_in"), py::arg("fan_out"), py::arg("seed") = 42)
        .def("forward", &Linear::forward)
        .def("parameters", &Linear::parameters);

    m.def("make_val", &make_val, py::arg("val"), py::arg("requires_grad") = true, "helper function to generate a shared_ptr node");

    py::class_<MLP>(m, "MLP")
        .def(py::init<int, std::vector<int>, std::string, int>(), 
             py::arg("fan_in"), 
             py::arg("hidden_sizes"), 
             py::arg("activation_layer") = "", 
             py::arg("seed") = 42)
        .def("forward", &MLP::forward)
        .def("parameters", &MLP::parameters);
    
    auto m_optim = m.def_submodule("optim", "optimization sub-algorithms manager");

    py::class_<SGD>(m_optim, "SGD")
        .def(py::init<std::vector<ValuePtr>, double, double, double>(),
             py::arg("params"),
             py::arg("lr"),
             py::arg("momentum") = 0.0,
             py::arg("weight_decay") = 0.0)
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad);
    
    py::class_<MSELoss>(m, "MSELoss")
        .def(py::init<>())
        .def("__call__", &MSELoss::operator());

    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("__call__", &CrossEntropyLoss::operator());
}