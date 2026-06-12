#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Value.h"
#include "nn/Linear.h"

namespace py = pybind11;

PYBIND11_MODULE(autograd_cpp, m) {
    m.doc() = "C++ Autograd Engine Python Bindings";

    py::class_<Value, std::shared_ptr<Value>>(m, "Value")
        .def_readwrite("data", &Value::data)
        .def_readwrite("grad", &Value::grad)
        .def_readwrite("op", &Value::op)
        
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

    m.def("make_val", &make_val, "Helper function to generate a shared_ptr node");
}