#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "autograd/Tensor.h"
#include "nn/Module.h"
#include "nn/Linear.h"
#include "nn/Conv2D.h"
#include "nn/MaxPool2D.h"
#include "nn/Sequential.h"
#include "nn/MLP.h"
#include "optim/SGD.h"
#include "optim/Adam.h"
#include "nn/Loss.h"
#include "utils/RNG.h"
#include "nn/BatchNorm2D.h"

namespace py = pybind11;

// ----------------------------------------------------
// PYBIND11 TRAMPOLINE CLASS FOR MODULE SUBCLASSING
// ----------------------------------------------------
class PyModule : public Module {
public:
    using Module::Module;

    TensorPtr forward(const TensorPtr& input) override {
        PYBIND11_OVERRIDE_PURE(
            TensorPtr,
            Module,
            forward,
            input
        );
    }

    std::vector<TensorPtr> parameters() const override {
        PYBIND11_OVERRIDE(
            std::vector<TensorPtr>,
            Module,
            parameters
        );
    }
};

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
            [](const TensorPtr& t) { 
                // allocate zero-filled gradient buffer if it does not exist yet
                t->ensure_grad_allocated(); 
                return *t->grad; 
            },
            [](TensorPtr& t, std::vector<double> v) { 
                // allocate gradient buffer before copying values from python list
                t->ensure_grad_allocated(); 
                *t->grad = v; 
                }
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
    // BASE MODULE BINDING
    // ----------------------------------------------------
    py::class_<Module, PyModule, std::shared_ptr<Module>>(m, "Module")
        .def(py::init<>())
        .def("forward", &Module::forward)
        .def("parameters", &Module::parameters)
        .def("zero_grad", &Module::zero_grad)
        .def("__setattr__", [](py::object self, const std::string& name, py::object value) {
            if (py::isinstance<Module>(value)) {
                auto native_self = self.cast<std::shared_ptr<Module>>();
                auto native_mod = value.cast<std::shared_ptr<Module>>();
                
                native_self->register_submodule(native_mod);
            }
            
            auto builtins = py::module_::import("builtins");
            auto object_setattr = builtins.attr("object").attr("__setattr__");
            object_setattr(self, name, value);
        });
    
    // ----------------------------------------------------
    // NEURAL NETWORK BINDINGS
    // ----------------------------------------------------
    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int, const std::string&>(), 
             py::arg("fan_in"), py::arg("fan_out"), py::arg("init_type") = "kaiming");

    py::class_<Conv2D, Module, std::shared_ptr<Conv2D>>(m, "Conv2D")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"), 
             py::arg("stride") = 1, py::arg("padding") = 0);

    py::class_<MaxPool2D, Module, std::shared_ptr<MaxPool2D>>(m, "MaxPool2D")
        .def(py::init<size_t, size_t>(),
             py::arg("kernel_size"), py::arg("stride") = 2);

    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(py::init<>())
        .def("add", &Sequential::add, py::arg("layer"))
        .def("__call__", [](Sequential& self, const TensorPtr& input) { return self.forward(input); });

    py::class_<MLP, Module, std::shared_ptr<MLP>>(m, "MLP")
        .def(py::init<int, std::vector<int>, std::string>(), 
             py::arg("fan_in"), 
             py::arg("hidden_sizes"), 
             py::arg("activation_layer") = "");
            
    py::class_<BatchNorm2D, Module, std::shared_ptr<BatchNorm2D>>(m, "BatchNorm2D")
        .def(py::init<size_t, double, double, bool, bool>(),
             py::arg("num_features"), py::arg("eps") = 1e-5, py::arg("momentum") = 0.1,
             py::arg("affine") = true, py::arg("track_running_stats") = true);
    
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
    
    py::class_<Adam>(m_optim, "Adam")
        .def(py::init<std::vector<TensorPtr>, double, std::pair<double, double>, double, double, bool, bool>(),
             py::arg("params"),
             py::arg("lr") = 0.001,
             py::arg("betas") = std::make_pair(0.9, 0.999),
             py::arg("eps") = 1e-8,
             py::arg("weight_decay") = 0.0,
             py::arg("amsgrad") = false,
             py::arg("maximize") = false)
        .def("step", &Adam::step)
        .def("zero_grad", &Adam::zero_grad);
    
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