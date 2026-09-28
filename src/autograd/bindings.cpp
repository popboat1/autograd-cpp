#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
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
#include "nn/AvgPool2D.h"

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
    // DEVICE ENUM BINDING
    // ----------------------------------------------------
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();

    // ----------------------------------------------------
    // TENSOR BINDINGS
    // ----------------------------------------------------
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init([](py::array_t<double> values, std::vector<size_t> shape, bool requires_grad, Device device) {
            py::buffer_info info = values.request();
            double* ptr = static_cast<double*>(info.ptr);
            std::vector<double> v(ptr, ptr + info.size);
            return std::make_shared<Tensor>(v, shape, requires_grad, device);
        }), py::arg("values"), py::arg("shape"), py::arg("requires_grad") = true, py::arg("device") = Device::CPU)
        .def_property("data",
            [](const TensorPtr& t) -> py::array_t<double> { 
                if (t->device == Device::CUDA && t->cuda_data != nullptr) {
                    CUDA_CHECK(cudaMemcpy(t->data->data(), t->cuda_data.get(), t->data->size() * sizeof(double), cudaMemcpyDeviceToHost));
                }
                return py::array_t<double>(t->data->size(), t->data->data(), py::cast(t)); 
            }, // getter
            [](TensorPtr& t, py::array_t<double> v) { 
                py::buffer_info info = v.request();
                double* ptr = static_cast<double*>(info.ptr);
                std::memcpy(t->data->data(), ptr, info.size * sizeof(double));
                if (t->device == Device::CUDA && t->cuda_data != nullptr) {
                    CUDA_CHECK(cudaMemcpy(t->cuda_data.get(), ptr, info.size * sizeof(double), cudaMemcpyHostToDevice));
                }
            } // setter
        )
        .def_property("grad",
            [](const TensorPtr& t) -> py::array_t<double> { 
                // allocate zero-filled gradient buffer if it does not exist yet
                t->ensure_grad_allocated(); 
                if (t->device == Device::CUDA && t->cuda_grad != nullptr) {
                    CUDA_CHECK(cudaMemcpy(t->grad->data(), t->cuda_grad.get(), t->grad->size() * sizeof(double), cudaMemcpyDeviceToHost));
                }
                return py::array_t<double>(t->grad->size(), t->grad->data(), py::cast(t)); 
            },
            [](TensorPtr& t, py::array_t<double> v) { 
                // allocate gradient buffer before copying values from numpy array
                t->ensure_grad_allocated(); 
                py::buffer_info info = v.request();
                double* ptr = static_cast<double*>(info.ptr);
                std::memcpy(t->grad->data(), ptr, info.size * sizeof(double));
                if (t->device == Device::CUDA && t->cuda_grad != nullptr) {
                    CUDA_CHECK(cudaMemcpy(t->cuda_grad.get(), ptr, info.size * sizeof(double), cudaMemcpyHostToDevice));
                }
            }
        )
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def_readonly("device", &Tensor::device)
        .def_property_readonly("requires_grad", [](const TensorPtr& self) { return self->requires_grad; })
        
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad) // allows manual gradient clearing on explicit nodes
        .def("to", &Tensor::to, py::arg("device"))
        .def("item", [](const TensorPtr& t) -> double {
            if (t->data->size() != 1) {
                throw std::runtime_error("item() is only valid for tensors with exactly 1 element");
            }
            if (t->device == Device::CUDA && t->cuda_data != nullptr) {
                double val = 0.0;
                CUDA_CHECK(cudaMemcpy(&val, t->cuda_data.get(), sizeof(double), cudaMemcpyDeviceToHost));
                return val;
            }
            return (*t->data)[0];
        })
        
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
        .def("to", &Module::to, py::arg("device"))
        .def("__call__", [](Module& self, const TensorPtr& input) { return self.forward(input); })
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
             py::arg("fan_in"), py::arg("fan_out"), py::arg("init_type") = "kaiming")
        .def_property_readonly("weights", [](const std::shared_ptr<Linear>& l) { return l->parameters()[0]; })
        .def_property_readonly("biases", [](const std::shared_ptr<Linear>& l) { return l->parameters()[1]; });

    py::class_<Conv2D, Module, std::shared_ptr<Conv2D>>(m, "Conv2D")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"), 
             py::arg("stride") = 1, py::arg("padding") = 0)
        .def_readonly("weight", &Conv2D::weight)
        .def_readonly("bias", &Conv2D::bias);

    py::class_<MaxPool2D, Module, std::shared_ptr<MaxPool2D>>(m, "MaxPool2D")
        .def(py::init<size_t, size_t>(),
             py::arg("kernel_size"), py::arg("stride") = 2);
    
    py::class_<AvgPool2D, Module, std::shared_ptr<AvgPool2D>>(m, "AveragePool2D")
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
             py::arg("affine") = true, py::arg("track_running_stats") = true)
        .def_readwrite("training", &BatchNorm2D::training)
        .def_readonly("running_mean", &BatchNorm2D::running_mean)
        .def_readonly("running_var", &BatchNorm2D::running_var)
        .def_readonly("weight", &BatchNorm2D::weight)
        .def_readonly("bias", &BatchNorm2D::bias);
    
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