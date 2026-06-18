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
    // let's build a basic 2x3 weight leaf matrix
    std::vector<double> a_vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<size_t> a_shape = {2, 3};
    auto a = std::make_shared<Tensor>(a_vals, a_shape, true);

    // shape coordinates check
    assert(a->shape[0] == 2);
    assert(a->shape[1] == 3);

    // a row major 2x3 matrix must calculate strides of [3, 1]
    // row step skips 3 items, column step skips 1 item
    assert(a->strides[0] == 3);
    assert(a->strides[1] == 1);

    // check flat lookup coordination mapping coordinates (row=1, col=1) -> index 4
    assert(a->get_flat_index({1, 1}) == 4);
    std::cout << "leaf instantiation, stride mapping, and tracking setups look clean\n";


    // test 2: verifying zero copy matrix transpositions
    // transposing turns our 2x3 matrix into a 3x2 matrix instantly
    auto a_t = a->transpose();

    assert(a_t->shape[0] == 3);
    assert(a_t->shape[1] == 2);

    // strides should flip completely to [1, 3] without shuffling raw data blocks around
    assert(a_t->strides[0] == 1);
    assert(a_t->strides[1] == 3);

    // verifying coordinate alignment values via mapped strides
    // original a(1,0) was 4.0. transposed a_t(0,1) must map to the same data index 4
    assert(a_t->data[a_t->get_flat_index({0, 1})] == 4.0);
    std::cout << "zero copy metadata tensor transposition verified successfully\n";


    // test 3: checking 2d matrix multiplication forward pass transformations
    // let's multiply a 2x2 matrix x by a 2x2 matrix w
    // x = [[1, 2],
    //      [3, 4]]
    // w = [[2, 0],
    //      [1, 2]]
    auto x = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    auto w = std::make_shared<Tensor>(std::vector<double>{2.0, 0.0, 1.0, 2.0}, std::vector<size_t>{2, 2}, true);

    // forward pass operation tracking line
    auto out = Tensor::matmul(x, w);

    // out shape must evaluate to 2x2
    assert(out->shape[0] == 2);
    assert(out->shape[1] == 2);

    // mathematical product validation values:
    // out = [[1*2 + 2*1, 1*0 + 2*2],  =  [[4, 4],
    //        [3*2 + 4*1, 3*0 + 4*2]]     [10, 8]]
    assert(close_enough(out->data[0], 4.0));
    assert(close_enough(out->data[1], 4.0));
    assert(close_enough(out->data[2], 10.0));
    assert(close_enough(out->data[3], 8.0));
    std::cout << "matrix multiplication forward pass outputs match analytical baselines\n";


    // test 4: checking autograd execution and joint matmul backpropagation paths
    // trigger reverse mode autodiff across the computed graph blocks
    out->backward();

    // let's check input node gradients against manual calculus evaluations
    // upstream out gradient initializes to [[1, 1], [1, 1]] via backward step
    // dx = dout * w^t
    // dx = [[1, 1],   * [[2, 1],  =  [[2, 3],
    //       [1, 1]]       [0, 2]]      [2, 3]]
    assert(close_enough(x->grad[0], 2.0));
    assert(close_enough(x->grad[1], 3.0));
    assert(close_enough(x->grad[2], 2.0));
    assert(close_enough(x->grad[3], 3.0));

    // dw = x^t * dout
    // dw = [[1, 3],   * [[1, 1],  =  [[4, 4],
    //       [2, 4]]       [1, 1]]      [6, 6]]
    assert(close_enough(w->grad[0], 4.0));
    assert(close_enough(w->grad[1], 4.0));
    assert(close_enough(w->grad[2], 6.0));
    assert(close_enough(w->grad[3], 6.0));

    std::cout << "reverse mode automatic differentiation gradients verified successfully\n";

    // test 5: checking element-wise addition forward and backward passes
    auto mat1 = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    auto mat2 = std::make_shared<Tensor>(std::vector<double>{5.0, 6.0, 7.0, 8.0}, std::vector<size_t>{2, 2}, true);

    auto add_out = mat1 + mat2;
    assert(close_enough(add_out->data[0], 6.0));
    assert(close_enough(add_out->data[3], 12.0));

    add_out->backward();
    assert(close_enough(mat1->grad[0], 1.0));
    assert(close_enough(mat2->grad[3], 1.0));
    std::cout << "element-wise addition forward and backward passes verified successfully\n";


    // test 6: checking element-wise subtraction forward and backward passes
    auto mat3 = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    auto mat4 = std::make_shared<Tensor>(std::vector<double>{5.0, 6.0, 7.0, 8.0}, std::vector<size_t>{2, 2}, true);

    auto sub_out = mat3 - mat4;
    assert(close_enough(sub_out->data[0], -4.0));
    assert(close_enough(sub_out->data[3], -4.0));

    sub_out->backward();
    assert(close_enough(mat3->grad[0], 1.0));
    assert(close_enough(mat4->grad[3], -1.0));
    std::cout << "element-wise subtraction forward and backward passes verified successfully\n";

    std::cout << "all tests passed cleanly\n";

    return 0;
}