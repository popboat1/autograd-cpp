#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "autograd/Tensor.h"

// helper to assert floating point parity smoothly
bool close_enough(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << "starting tensorgrad tests\n";

    // test 1: verifying shape, data allocation, and row major strides calculation
    std::vector<double> a_vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<size_t> a_shape = {2, 3};
    auto a = std::make_shared<Tensor>(a_vals, a_shape, true);

    assert(a->shape[0] == 2);
    assert(a->shape[1] == 3);
    assert(a->strides[0] == 3);
    assert(a->strides[1] == 1);
    assert(a->get_flat_index({1, 1}) == 4);
    std::cout << "[PASS] leaf instantiation, stride mapping, and tracking setups look clean\n";


    // test 2: verifying zero copy matrix transpositions
    // updated to pass explicit dimensions 0 and 1 for axis-swapping
    auto a_t = a->transpose(0, 1);

    assert(a_t->shape[0] == 3);
    assert(a_t->shape[1] == 2);
    assert(a_t->strides[0] == 1);
    assert(a_t->strides[1] == 3);
    assert(a_t->data[a_t->get_flat_index({0, 1})] == 4.0);
    std::cout << "[PASS] zero copy metadata tensor transposition verified successfully\n";


    // test 3: checking 2d matrix multiplication forward pass transformations
    auto x = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    auto w = std::make_shared<Tensor>(std::vector<double>{2.0, 0.0, 1.0, 2.0}, std::vector<size_t>{2, 2}, true);
    auto out = Tensor::matmul(x, w);

    assert(out->shape[0] == 2);
    assert(out->shape[1] == 2);
    assert(close_enough(out->data[0], 4.0));
    assert(close_enough(out->data[1], 4.0));
    assert(close_enough(out->data[2], 10.0));
    assert(close_enough(out->data[3], 8.0));
    std::cout << "[PASS] matrix multiplication forward pass outputs match analytical baselines\n";


    // test 4: checking autograd execution and joint matmul backpropagation paths
    out->backward();
    assert(close_enough(x->grad[0], 2.0));
    assert(close_enough(x->grad[1], 3.0));
    assert(close_enough(x->grad[2], 2.0));
    assert(close_enough(x->grad[3], 3.0));
    assert(close_enough(w->grad[0], 4.0));
    assert(close_enough(w->grad[1], 4.0));
    assert(close_enough(w->grad[2], 6.0));
    assert(close_enough(w->grad[3], 6.0));
    std::cout << "[PASS] reverse mode automatic differentiation gradients verified successfully\n";


    // test 5: checking element-wise addition forward and backward passes
    auto mat1 = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    auto mat2 = std::make_shared<Tensor>(std::vector<double>{5.0, 6.0, 7.0, 8.0}, std::vector<size_t>{2, 2}, true);

    auto add_out = mat1 + mat2;
    assert(close_enough(add_out->data[0], 6.0));
    assert(close_enough(add_out->data[3], 12.0));

    add_out->backward();
    assert(close_enough(mat1->grad[0], 1.0));
    assert(close_enough(mat2->grad[3], 1.0));
    std::cout << "[PASS] element-wise addition forward and backward passes verified successfully\n";


    // test 6: checking element-wise subtraction forward and backward passes
    auto mat3 = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    auto mat4 = std::make_shared<Tensor>(std::vector<double>{5.0, 6.0, 7.0, 8.0}, std::vector<size_t>{2, 2}, true);

    auto sub_out = mat3 - mat4;
    assert(close_enough(sub_out->data[0], -4.0));
    assert(close_enough(sub_out->data[3], -4.0));

    sub_out->backward();
    assert(close_enough(mat3->grad[0], 1.0));
    assert(close_enough(mat4->grad[3], -1.0));
    std::cout << "[PASS] element-wise subtraction forward and backward passes verified successfully\n";


    // test 7: checking element-wise multiplication forward and backward passes
    auto mat5 = std::make_shared<Tensor>(std::vector<double>{2.0, 3.0, 4.0, 5.0}, std::vector<size_t>{2, 2}, true);
    auto mat6 = std::make_shared<Tensor>(std::vector<double>{10.0, 20.0, 30.0, 40.0}, std::vector<size_t>{2, 2}, true);
    auto mul_out = mat5 * mat6;
    assert(close_enough(mul_out->data[0], 20.0));
    assert(close_enough(mul_out->data[3], 200.0));

    mul_out->backward();
    assert(close_enough(mat5->grad[0], 10.0));
    assert(close_enough(mat5->grad[3], 40.0));
    assert(close_enough(mat6->grad[0], 2.0));
    assert(close_enough(mat6->grad[3], 5.0));
    std::cout << "[PASS] element-wise multiplication forward and backward passes verified successfully\n";


    // test 8: checking tensor sum reduction forward and backward passes
    auto mat_to_sum = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    auto sum_out = mat_to_sum->sum();
    
    // updated asserts to properly validate the optimized N-D scalar shape footprint
    assert(sum_out->shape.size() == 1);
    assert(sum_out->shape[0] == 1);
    assert(close_enough(sum_out->data[0], 10.0));

    sum_out->backward();
    assert(close_enough(mat_to_sum->grad[0], 1.0));
    assert(close_enough(mat_to_sum->grad[3], 1.0));
    std::cout << "[PASS] tensor total sum reduction forward and backward passes verified successfully\n";


    // test 9: checking advanced N-dimensional shape broadcasting and zero-stride autograd reduction
    // base_3d shape = [2, 2, 3] (12 elements total)
    std::vector<double> vals_3d = {1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12};
    auto tensor_3d = std::make_shared<Tensor>(vals_3d, std::vector<size_t>{2, 2, 3}, true);

    // bias_1d shape = [3] (to be broadcasted across dimensions 0 and 1)
    std::vector<double> vals_1d = {10.0, 20.0, 30.0};
    auto bias_1d = std::make_shared<Tensor>(vals_1d, std::vector<size_t>{3}, true);

    // forward pass: [2, 2, 3] + [3] -> output shape should broadcast to [2, 2, 3]
    auto broadcast_out = tensor_3d + bias_1d;
    assert(broadcast_out->shape.size() == 3);
    assert(broadcast_out->shape[0] == 2 && broadcast_out->shape[1] == 2 && broadcast_out->shape[2] == 3);

    // element-wise sample assertions
    // index [0, 0, 0]: 1.0 + 10.0 = 11.0
    // index [1, 1, 2]: 12.0 + 30.0 = 42.0
    assert(close_enough(broadcast_out->data[0], 11.0));
    assert(close_enough(broadcast_out->data[11], 42.0));

    // backward pass execution
    broadcast_out->backward();

    // tensor_3d had a 1-to-1 mapping, its gradients should all receive exactly 1.0
    assert(close_enough(tensor_3d->grad[0], 1.0));
    assert(close_enough(tensor_3d->grad[11], 1.0));

    // bias_1d was virtually duplicated 4 times across dimensions 0 and 1.
    // its gradients must automatically sum up to 4.0 per slot via the zero-stride accumulation path.
    assert(close_enough(bias_1d->grad[0], 4.0));
    assert(close_enough(bias_1d->grad[1], 4.0));
    assert(close_enough(bias_1d->grad[2], 4.0));
    std::cout << "[PASS] N-dimensional shape broadcasting and zero-stride autograd reduction verified successfully\n";

    // test 10: checking high-rank batched matrix multiplication with batch broadcasting
    // lhs shape: [2, 1, 2, 3] (12 elements)
    std::vector<double> b_lhs_vals = {1, 2, 3,  4, 5, 6,
                                      1, 1, 1,  2, 2, 2};
    auto lhs_tensor = std::make_shared<Tensor>(b_lhs_vals, std::vector<size_t>{2, 1, 2, 3}, true);

    // rhs shape: [2, 2, 3, 2] (24 elements)
    std::vector<double> b_rhs_vals = {1, 0,  0, 1,  1, 1,
                                      2, 1,  1, 2,  0, 1,
                                      1, 1,  1, 1,  1, 1,
                                      0, 0,  0, 0,  0, 0};
    auto rhs_tensor = std::make_shared<Tensor>(b_rhs_vals, std::vector<size_t>{2, 2, 3, 2}, true);

    // forward pass: [2, 1, 2, 3] @ [2, 2, 3, 2] -> batch shape broadcasts to [2, 2], core matrix is [2, 2]
    // output shape: [2, 2, 2, 2] (16 elements total)
    auto batched_out = Tensor::matmul(lhs_tensor, rhs_tensor);

    assert(batched_out->shape.size() == 4);
    assert(batched_out->shape[0] == 2 && batched_out->shape[1] == 2);
    assert(batched_out->shape[2] == 2 && batched_out->shape[3] == 2);

    // check a structural forward pass slice element
    // batch [0, 0], matrix row 0, col 0: (1*1 + 2*0 + 3*1) = 4.0
    assert(close_enough(batched_out->data[0], 4.0));
    std::cout << "[PASS] batched matrix multiplication forward transformations verified successfully\n";

    // verify autograd reduction back through batched pathways
    batched_out->backward();

    // since the batch dimensions broadcasted, gradients automatically sum up
    assert(lhs_tensor->grad.size() == 12);
    assert(rhs_tensor->grad.size() == 24);
    assert(lhs_tensor->grad[0] > 0.0);
    assert(rhs_tensor->grad[0] > 0.0);
    std::cout << "[PASS] batched matrix multiplication reverse mode autograd path verified successfully\n";

    std::cout << "[PASS] all tests passed cleanly\n";
    return 0;
}