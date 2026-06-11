#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Value.h"

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
        .def("__truediv__", [](const ValuePtr& lhs, const ValuePtr& rhs) { return lhs / rhs; });

    m.def("make_val", &make_val, "Helper function to generate a shared_ptr node");
}