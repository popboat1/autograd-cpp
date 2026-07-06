#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "autograd/Tensor.h"
#include "nn/Linear.h"
#include "nn/MLP.h"
#include "optim/SGD.h"
#include "nn/Loss.h"
#include "utils/RNG.h"

namespace py = pybind11;

PYBIND11_MODULE(autograd_cpp, m) {
    m.doc() = "C++ ML Framework Python Bindings (Tensor Engine)";

    // ----------------------------------------------------
    // GLOBAL UTILITIES & STOCHASTICITY
    // ----------------------------------------------------
    m.def("manual_seed", &RNG::manual_seed, py::arg("seed"), "Set global framework seed");

    // ----------------------------------------------------
    // TENSOR BINDINGS
    // ----------------------------------------------------
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<double>, std::vector<size_t>, bool>(), 
             py::arg("values"), py::arg("shape"), py::arg("requires_grad") = true)
        .def_property("data",
            [](const TensorPtr& t) { return *t->data; }, // getter
            [](TensorPtr& t, std::vector<double> v) { *t->data = v; } // setter
        )
        .def_property("grad",
            [](const TensorPtr& t) { return *t->grad; },
            [](TensorPtr& t, std::vector<double> v) { *t->grad = v; }
        )
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def_property_readonly("requires_grad", [](const TensorPtr& self) { return self->requires_grad; })
        
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad) // allows manual gradient clearing on explicit nodes
        
        // operations
        .def("sum", py::overload_cast<>(&Tensor::sum))
        .def("sum", py::overload_cast<size_t, bool>(&Tensor::sum), 
             py::arg("dim"), py::arg("keepdim") = false)
        .def("mean", &Tensor::mean, py::arg("dim"), py::arg("keepdim") = false)
        .def("max", &Tensor::max, py::arg("dim"), py::arg("keepdim") = false)
        .def("argmax", &Tensor::argmax, py::arg("dim"), py::arg("keepdim") = false)
        .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
        .def("view", &Tensor::view, py::arg("target_shape"))

        // shape manipulation additions
        .def("reshape", &Tensor::reshape, py::arg("new_shape"))
        .def("squeeze", &Tensor::squeeze, py::arg("dim"))
        .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim"))
        .def("permute", &Tensor::permute, py::arg("dims"))
        
        // math/activations
        .def("pow", &Tensor::pow)
        .def("tanh", &Tensor::tanh)
        .def("exp", &Tensor::exp)
        .def("relu", &Tensor::relu)
        .def("sigmoid", &Tensor::sigmoid)
        .def("log", &Tensor::log)
        .def("print", &Tensor::print)

        // advanced activations
        .def("softmax", &Tensor::softmax, py::arg("dim"))
        .def("log_softmax", &Tensor::log_softmax, py::arg("dim"))

        // magic methods
        .def("__add__", [](const TensorPtr& lhs, const TensorPtr& rhs) { return lhs + rhs; })
        .def("__sub__", [](const TensorPtr& lhs, const TensorPtr& rhs) { return lhs - rhs; })
        .def("__mul__", [](const TensorPtr& lhs, const TensorPtr& rhs) { return lhs * rhs; })
        .def("__truediv__", [](const TensorPtr& lhs, const TensorPtr& rhs) { return lhs / rhs; })
        
        // map matmul to @ operator
        .def("__matmul__", [](const TensorPtr& lhs, const TensorPtr& rhs) { return Tensor::matmul(lhs, rhs); })
        
        // unary ops
        .def("sqrt", &Tensor::sqrt)
        .def("neg", &Tensor::neg)
        .def("__neg__", [](const TensorPtr& self) { return -self; })

        .def("expand", &Tensor::expand, py::arg("new_shape"))

        .def("argsort", &Tensor::argsort, py::arg("dim"), py::arg("descending") = false)

        // comparison Operator Bindings
        .def("__eq__", [](const TensorPtr& lhs, const TensorPtr& rhs) { return *lhs == *rhs; })
        .def("__lt__", [](const TensorPtr& lhs, const TensorPtr& rhs) { return *lhs < *rhs; })
        .def("__gt__", [](const TensorPtr& lhs, const TensorPtr& rhs) { return *lhs > *rhs; });
    
    // ----------------------------------------------------
    // NEURAL NETWORK BINDINGS
    // ----------------------------------------------------
    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int, const std::string&>(), 
             py::arg("fan_in"), py::arg("fan_out"), py::arg("init_type") = "kaiming")
        .def("forward", &Linear::forward)
        .def("parameters", &Linear::parameters);

    py::class_<MLP>(m, "MLP")
        .def(py::init<int, std::vector<int>, std::string>(), 
             py::arg("fan_in"), 
             py::arg("hidden_sizes"), 
             py::arg("activation_layer") = "")
        .def("forward", &MLP::forward)
        .def("parameters", &MLP::parameters);
    
    // ----------------------------------------------------
    // OPTIMIZER & LOSS BINDINGS
    // ----------------------------------------------------
    auto m_optim = m.def_submodule("optim", "optimization sub-algorithms manager");

    py::class_<SGD>(m_optim, "SGD")
        .def(py::init<std::vector<TensorPtr>, double, double, double>(),
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

    py::class_<SparseCategoricalCrossEntropyLoss>(m, "SparseCategoricalCrossEntropyLoss")
        .def(py::init<>())
        .def("__call__", &SparseCategoricalCrossEntropyLoss::operator());
}