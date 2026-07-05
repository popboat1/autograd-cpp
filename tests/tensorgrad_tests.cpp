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


    // test 2: verifying zero copy metadata tensor transposition
    auto a_t = a->transpose(0, 1);

    assert(a_t->shape[0] == 3);
    assert(a_t->shape[1] == 2);
    assert(a_t->strides[0] == 1);
    assert(a_t->strides[1] == 3);
    assert((*a_t->data)[a_t->get_flat_index({0, 1})] == 4.0);
    std::cout << "[PASS] zero copy metadata tensor transposition verified successfully\n";


    // test 3: checking 2d matrix multiplication forward pass transformations
    auto x = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    auto w = std::make_shared<Tensor>(std::vector<double>{2.0, 0.0, 1.0, 2.0}, std::vector<size_t>{2, 2}, true);
    auto out = Tensor::matmul(x, w);

    assert(out->shape[0] == 2);
    assert(out->shape[1] == 2);
    assert(close_enough((*out->data)[0], 4.0));
    assert(close_enough((*out->data)[1], 4.0));
    assert(close_enough((*out->data)[2], 10.0));
    assert(close_enough((*out->data)[3], 8.0));
    std::cout << "[PASS] matrix multiplication forward pass outputs match analytical baselines\n";


    // test 4: checking autograd execution and joint matmul backpropagation paths
    out->backward();
    assert(close_enough((*x->grad)[0], 2.0));
    assert(close_enough((*x->grad)[1], 3.0));
    assert(close_enough((*x->grad)[2], 2.0));
    assert(close_enough((*x->grad)[3], 3.0));
    assert(close_enough((*w->grad)[0], 4.0));
    assert(close_enough((*w->grad)[1], 4.0));
    assert(close_enough((*w->grad)[2], 6.0));
    assert(close_enough((*w->grad)[3], 6.0));
    std::cout << "[PASS] reverse mode automatic differentiation gradients verified successfully\n";


    // test 5: checking element-wise addition forward and backward passes
    auto mat1 = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    auto mat2 = std::make_shared<Tensor>(std::vector<double>{5.0, 6.0, 7.0, 8.0}, std::vector<size_t>{2, 2}, true);

    auto add_out = mat1 + mat2;
    assert(close_enough((*add_out->data)[0], 6.0));
    assert(close_enough((*add_out->data)[3], 12.0));

    add_out->backward();
    assert(close_enough((*mat1->grad)[0], 1.0));
    assert(close_enough((*mat2->grad)[3], 1.0));
    std::cout << "[PASS] element-wise addition forward and backward passes verified successfully\n";


    // test 6: checking element-wise subtraction forward and backward passes
    auto mat3 = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    auto mat4 = std::make_shared<Tensor>(std::vector<double>{5.0, 6.0, 7.0, 8.0}, std::vector<size_t>{2, 2}, true);

    auto sub_out = mat3 - mat4;
    assert(close_enough((*sub_out->data)[0], -4.0));
    assert(close_enough((*sub_out->data)[3], -4.0));

    sub_out->backward();
    assert(close_enough((*mat3->grad)[0], 1.0));
    assert(close_enough((*mat4->grad)[3], -1.0));
    std::cout << "[PASS] element-wise subtraction forward and backward passes verified successfully\n";


    // test 7: checking element-wise multiplication forward and backward passes
    auto mat5 = std::make_shared<Tensor>(std::vector<double>{2.0, 3.0, 4.0, 5.0}, std::vector<size_t>{2, 2}, true);
    auto mat6 = std::make_shared<Tensor>(std::vector<double>{10.0, 20.0, 30.0, 40.0}, std::vector<size_t>{2, 2}, true);
    auto mul_out = mat5 * mat6;
    assert(close_enough((*mul_out->data)[0], 20.0));
    assert(close_enough((*mul_out->data)[3], 200.0));

    mul_out->backward();
    assert(close_enough((*mat5->grad)[0], 10.0));
    assert(close_enough((*mat5->grad)[3], 40.0));
    assert(close_enough((*mat6->grad)[0], 2.0));
    assert(close_enough((*mat6->grad)[3], 5.0));
    std::cout << "[PASS] element-wise multiplication forward and backward passes verified successfully\n";


    // test 8: checking tensor sum reduction along a specific dimension (forward and backward passes)  
    auto mat_to_sum = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, true);
    
    // Sum along dimension 0, keepdim = true    
    auto sum_out_kd = mat_to_sum->sum(0, true);
    assert(sum_out_kd->shape.size() == 2);
    assert(sum_out_kd->shape[0] == 1 && sum_out_kd->shape[1] == 2);
    assert(close_enough((*sum_out_kd->data)[0], 4.0));   
    assert(close_enough((*sum_out_kd->data)[1], 6.0));    
    
    // Sum along dimension 1, keepdim = false
    auto sum_out = mat_to_sum->sum(1, false);
    assert(sum_out->shape.size() == 1);
    assert(sum_out->shape[0] == 2);
    assert(close_enough((*sum_out->data)[0], 3.0));
    assert(close_enough((*sum_out->data)[1], 7.0));
    
    sum_out->backward();
    assert(close_enough((*mat_to_sum->grad)[0], 1.0));
    assert(close_enough((*mat_to_sum->grad)[1], 1.0));
    assert(close_enough((*mat_to_sum->grad)[2], 1.0));
    assert(close_enough((*mat_to_sum->grad)[3], 1.0));
    
    std::cout << "[PASS] tensor dimensional sum reduction forward and backward passes verified successfully\n";

    // test 9: checking advanced N-dimensional shape broadcasting and zero-stride autograd reduction
    std::vector<double> vals_3d = {1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12};
    auto tensor_3d = std::make_shared<Tensor>(vals_3d, std::vector<size_t>{2, 2, 3}, true);

    std::vector<double> vals_1d = {10.0, 20.0, 30.0};
    auto bias_1d = std::make_shared<Tensor>(vals_1d, std::vector<size_t>{3}, true);

    auto broadcast_out = tensor_3d + bias_1d;
    assert(broadcast_out->shape.size() == 3);
    assert(broadcast_out->shape[0] == 2 && broadcast_out->shape[1] == 2 && broadcast_out->shape[2] == 3);

    assert(close_enough((*broadcast_out->data)[0], 11.0));
    assert(close_enough((*broadcast_out->data)[11], 42.0));

    broadcast_out->backward();

    assert(close_enough((*tensor_3d->grad)[0], 1.0));
    assert(close_enough((*tensor_3d->grad)[11], 1.0));

    assert(close_enough((*bias_1d->grad)[0], 4.0));
    assert(close_enough((*bias_1d->grad)[1], 4.0));
    assert(close_enough((*bias_1d->grad)[2], 4.0));
    std::cout << "[PASS] N-dimensional shape broadcasting and zero-stride autograd reduction verified successfully\n";

    // test 10: checking high-rank batched matrix multiplication with batch broadcasting
    std::vector<double> b_lhs_vals = {1, 2, 3,  4, 5, 6,
                                      1, 1, 1,  2, 2, 2};
    auto lhs_tensor = std::make_shared<Tensor>(b_lhs_vals, std::vector<size_t>{2, 1, 2, 3}, true);

    std::vector<double> b_rhs_vals = {1, 0,  0, 1,  1, 1,
                                      2, 1,  1, 2,  0, 1,
                                      1, 1,  1, 1,  1, 1,
                                      0, 0,  0, 0,  0, 0};
    auto rhs_tensor = std::make_shared<Tensor>(b_rhs_vals, std::vector<size_t>{2, 2, 3, 2}, true);

    auto batched_out = Tensor::matmul(lhs_tensor, rhs_tensor);

    assert(batched_out->shape.size() == 4);
    assert(batched_out->shape[0] == 2 && batched_out->shape[1] == 2);
    assert(batched_out->shape[2] == 2 && batched_out->shape[3] == 2);

    assert(close_enough((*batched_out->data)[0], 4.0));
    std::cout << "[PASS] batched matrix multiplication forward transformations verified successfully\n";

    batched_out->backward();

    assert(lhs_tensor->grad->size() == 12);
    assert(rhs_tensor->grad->size() == 24);
    assert((*lhs_tensor->grad)[0] > 0.0);
    assert((*rhs_tensor->grad)[0] > 0.0);
    std::cout << "[PASS] batched matrix multiplication reverse mode autograd path verified successfully\n";
    
    // test 11: checking tensor view manipulation and flat autograd remapping
    std::vector<double> base_vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto base_tensor = std::make_shared<Tensor>(base_vals, std::vector<size_t>{8}, true);

    auto viewed_tensor = base_tensor->view({2, -1});
    
    assert(viewed_tensor->shape.size() == 2);
    assert(viewed_tensor->shape[0] == 2 && viewed_tensor->shape[1] == 4);
    assert(viewed_tensor->strides[0] == 4 && viewed_tensor->strides[1] == 1);
    
    // verify the true zero-copy data sharing mechanism
    (*viewed_tensor->data)[5] = 99.0;
    assert(close_enough((*base_tensor->data)[5], 99.0));
    (*viewed_tensor->data)[5] = 6.0; // reset
    std::cout << "[PASS] true zero-copy tensor view transformations verified\n";

    // execute 1-to-1 backpropagation mapping
    viewed_tensor->backward();
    assert(base_tensor->grad->size() == 8);
    assert(close_enough((*base_tensor->grad)[0], 1.0));
    assert(close_enough((*base_tensor->grad)[7], 1.0));
    std::cout << "[PASS] zero-copy tensor view flat backward paths verified\n";


    // test 12: visual check for N-dimensional bracket printing formatting
    std::cout << "\n--- visual print verification stream ---\n";
    std::vector<double> p_vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto p_tensor = std::make_shared<Tensor>(p_vals, std::vector<size_t>{2, 2, 3});
    
    std::cout << "printing a [2, 2, 3] layout tensor structure:\n";
    p_tensor->print();
    std::cout << "----------------------------------------\n";

    // test 13: checking element-wise division and advanced math/activations
    auto d_num = std::make_shared<Tensor>(std::vector<double>{2.0, -4.0, 1.0}, std::vector<size_t>{3}, true);
    auto d_den = std::make_shared<Tensor>(std::vector<double>{2.0, 2.0, 0.5}, std::vector<size_t>{3}, true);

    // Test operator/
    auto div_out = d_num / d_den;
    assert(close_enough((*div_out->data)[0], 1.0));
    assert(close_enough((*div_out->data)[1], -2.0));
    assert(close_enough((*div_out->data)[2], 2.0));

    // Test ReLU
    auto relu_out = d_num->relu();
    assert(close_enough((*relu_out->data)[0], 2.0));
    assert(close_enough((*relu_out->data)[1], 0.0));
    assert(close_enough((*relu_out->data)[2], 1.0));

    // Check ReLU backward pass
    relu_out->backward();
    assert(close_enough((*d_num->grad)[0], 1.0));
    assert(close_enough((*d_num->grad)[1], 0.0));
    std::cout << "[PASS] element-wise division and math activations verified successfully\n";


    // test 14: checking dimensional reductions (mean, max, argmax) and keepdim
    std::vector<double> r_vals = {1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0};
    auto r_tensor = std::make_shared<Tensor>(r_vals, std::vector<size_t>{2, 3}, true);

    // test mean
    auto mean_dim0 = r_tensor->mean(0, true); 
    assert(mean_dim0->shape.size() == 2);
    assert(mean_dim0->shape[0] == 1 && mean_dim0->shape[1] == 3);
    assert(close_enough((*mean_dim0->data)[0], 2.5));
    assert(close_enough((*mean_dim0->data)[2], 4.5));

    // test max
    auto max_dim1 = r_tensor->max(1, false);
    assert(max_dim1->shape.size() == 1);
    assert(max_dim1->shape[0] == 2);
    assert(close_enough((*max_dim1->data)[0], 3.0));
    assert(close_enough((*max_dim1->data)[1], 6.0));

    // test argmax
    auto argmax_dim1 = r_tensor->argmax(1, false);
    assert(close_enough((*argmax_dim1->data)[0], 2.0));
    assert(close_enough((*argmax_dim1->data)[1], 2.0));
    assert(argmax_dim1->requires_grad == false);

    // verify routing
    max_dim1->backward();
    assert(close_enough((*r_tensor->grad)[2], 1.0));
    assert(close_enough((*r_tensor->grad)[5], 1.0));
    assert(close_enough((*r_tensor->grad)[0], 0.0));
    assert(close_enough((*r_tensor->grad)[3], 0.0));
    std::cout << "[PASS] dimensional reductions, keepdim broadcasting, and autograd routing verified\n";

    // test 15: checking tensor shape manipulations (reshape, squeeze, unsqueeze, permute)
    std::vector<double> sm_vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto sm_tensor = std::make_shared<Tensor>(sm_vals, std::vector<size_t>{2, 3}, true);

    // test reshape layout alignment properties
    auto reshaped = sm_tensor->reshape({3, 2});
    assert(reshaped->shape[0] == 3 && reshaped->shape[1] == 2);
    assert(reshaped->strides[0] == 2 && reshaped->strides[1] == 1);

    // test unsqueeze axis padding extension
    auto unsqueezed = reshaped->unsqueeze(1); // shape shifts from {3, 2} to {3, 1, 2}
    assert(unsqueezed->shape.size() == 3);
    assert(unsqueezed->shape[0] == 3 && unsqueezed->shape[1] == 1 && unsqueezed->shape[2] == 2);
    assert(unsqueezed->strides[0] == 2 && unsqueezed->strides[1] == 2 && unsqueezed->strides[2] == 1);

    // test squeeze axis elimination recovery
    auto squeezed = unsqueezed->squeeze(1); // shape shifts back from {3, 1, 2} to {3, 2}
    assert(squeezed->shape.size() == 2);
    assert(squeezed->shape[0] == 3 && squeezed->shape[1] == 2);
    assert(squeezed->strides[0] == 2 && squeezed->strides[1] == 1);

    // test complex high-rank axis permute routing
    auto permuted = squeezed->permute({1, 0}); // shape shifts from {3, 2} to {2, 3} (matrix transpose)
    assert(permuted->shape[0] == 2 && permuted->shape[1] == 3);
    assert(permuted->strides[0] == 1 && permuted->strides[1] == 2); // verifies stride inversion layout
    
    // validate coordinate indexing tracking sanity over the permuted view block
    // logical coordinate {1, 0} on a transposed matrix maps to value 2.0
    size_t flat_view_idx = permuted->get_flat_index({1, 0});
    assert((*permuted->data)[flat_view_idx] == 2.0);
    std::cout << "[PASS] shape manipulations (reshape, squeeze, unsqueeze, permute) verified successfully\n";

    // test 16: checking element-wise tensor comparison matrices (==, <, >) with broadcasting
    auto comp_lhs = std::make_shared<Tensor>(std::vector<double>{1.0, 5.0, 3.0, 8.0}, std::vector<size_t>{2, 2}, true);
    auto comp_rhs = std::make_shared<Tensor>(std::vector<double>{2.0, 5.0}, std::vector<size_t>{1, 2}, true); // Broadcasts down row 1

    // test operator==
    auto eq_out = *comp_lhs == *comp_rhs;
    assert(eq_out->requires_grad == false); // Enforces absolute non-differentiable tracking state
    assert(close_enough((*eq_out->data)[0], 0.0)); // 1.0 == 2.0 -> false
    assert(close_enough((*eq_out->data)[1], 1.0)); // 5.0 == 5.0 -> true
    assert(close_enough((*eq_out->data)[2], 0.0)); // 3.0 == 2.0 -> false
    assert(close_enough((*eq_out->data)[3], 0.0)); // 8.0 == 5.0 -> false

    // test operator<
    auto lt_out = *comp_lhs < *comp_rhs;
    assert(close_enough((*lt_out->data)[0], 1.0)); // 1.0 < 2.0 -> true
    assert(close_enough((*lt_out->data)[1], 0.0)); // 5.0 < 5.0 -> false

    // test operator>
    auto gt_out = *comp_lhs > *comp_rhs;
    assert(close_enough((*gt_out->data)[2], 1.0)); // 3.0 > 2.0 -> true
    assert(close_enough((*gt_out->data)[3], 1.0)); // 8.0 > 5.0 -> true
    std::cout << "[PASS] element-wise tensor comparisons (==, <, >) and broadcast assertions verified\n";

    // test 17: checking advanced unary operations (sqrt, neg, and operator-) with autograd tracking
    auto unary_base = std::make_shared<Tensor>(std::vector<double>{4.0, 16.0}, std::vector<size_t>{2}, true);

    // test Square Root
    auto sqrt_out = unary_base->sqrt();
    assert(close_enough((*sqrt_out->data)[0], 2.0));
    assert(close_enough((*sqrt_out->data)[1], 4.0));

    // test Prefix Negation
    auto neg_out = -unary_base;
    assert(close_enough((*neg_out->data)[0], -4.0));
    assert(close_enough((*neg_out->data)[1], -16.0));

    // verify backpropagation derivatives
    sqrt_out->backward(); // d(sqrt(x))/dx = 0.5 / sqrt(x)
    assert(close_enough((*unary_base->grad)[0], 0.25)); // 0.5 / 2.0 -> 0.25
    assert(close_enough((*unary_base->grad)[1], 0.125)); // 0.5 / 4.0 -> 0.125

    unary_base->zero_grad();
    neg_out->backward();
    assert(close_enough((*unary_base->grad)[0], -1.0));
    assert(close_enough((*unary_base->grad)[1], -1.0));
    std::cout << "[PASS] advanced unary operations (sqrt, neg) and reverse backpropagation verified\n";

    // test 18: checking tensor view expansion (expand) and broadcast gradient reduction passes
    std::vector<double> exp_vals = {2.0, 4.0}; 
    auto exp_base = std::make_shared<Tensor>(exp_vals, std::vector<size_t>{2, 1}, true); // Shape (2, 1)

    // expand singleton dimension 1 from size 1 to size 3 -> logical shape becomes (2, 3)
    auto expanded_tensor = exp_base->expand({2, 3});
    assert(expanded_tensor->shape[0] == 2 && expanded_tensor->shape[1] == 3);
    assert(expanded_tensor->strides[0] == 1 && expanded_tensor->strides[1] == 0); // Stride 0 confirms zero-copy trick

    // verify zero-copy layout access mapping
    assert(close_enough((*expanded_tensor->data)[expanded_tensor->get_flat_index({0, 0})], 2.0));
    assert(close_enough((*expanded_tensor->data)[expanded_tensor->get_flat_index({0, 1})], 2.0)); // Broadcasted element
    assert(close_enough((*expanded_tensor->data)[expanded_tensor->get_flat_index({0, 2})], 2.0)); // Broadcasted element
    assert(close_enough((*expanded_tensor->data)[expanded_tensor->get_flat_index({1, 2})], 4.0)); // Second row stretch

    // simulate an upstream gradient matrix injection of 1.0s across all elements
    std::fill(expanded_tensor->grad->begin(), expanded_tensor->grad->end(), 1.0);

    // trigger backward pass to verify reduction summation routing
    expanded_tensor->backward_func(); 
    // each row of size 3 collapse back into its original size 1 singleton location -> total gradient = 3.0 per row
    assert(close_enough((*exp_base->grad)[0], 3.0));
    assert(close_enough((*exp_base->grad)[1], 3.0));
    std::cout << "[PASS] tensor view expand forward structures and backward reduction tracking verified\n";

    // test 19: verifying tensor argsort utility along a target dimension
    std::vector<double> sort_vals = {3.0, 1.0, 2.0, 6.0, 5.0, 4.0};
    auto sort_tensor = std::make_shared<Tensor>(sort_vals, std::vector<size_t>{2, 3}, false);

    // sort along dimension 1 (rows), ascending
    auto argsort_out = sort_tensor->argsort(1, false);
    assert(argsort_out->shape[0] == 2 && argsort_out->shape[1] == 3);
    assert(argsort_out->requires_grad == false);

    // row 0: [3, 1, 2] -> sorted indices should be [1, 2, 0]
    assert(close_enough((*argsort_out->data)[0], 1.0));
    assert(close_enough((*argsort_out->data)[1], 2.0));
    assert(close_enough((*argsort_out->data)[2], 0.0));

    // row 1: [6, 5, 4] -> sorted indices should be [2, 1, 0]
    assert(close_enough((*argsort_out->data)[3], 2.0));
    assert(close_enough((*argsort_out->data)[4], 1.0));
    assert(close_enough((*argsort_out->data)[5], 0.0));
    std::cout << "[PASS] tensor argsort index generation utilities verified successfully\n";

    // test 20: verifying explicit zero_grad operation on intermediate graph nodes
    auto base_x = std::make_shared<Tensor>(std::vector<double>{2.0}, std::vector<size_t>{1}, true);
    auto intermediate_y = base_x * base_x; // y = x^2, dy/dx = 2x = 4.0
    auto final_z = intermediate_y * 3.0;   // z = 3y, dz/dy = 3.0

    // execute first forward/backward pass
    final_z->backward();
    assert(close_enough((*intermediate_y->grad)[0], 3.0));
    assert(close_enough((*base_x->grad)[0], 12.0)); // dz/dx = 3 * 2x = 12.0

    // verify that backward accumulates gradients by default if uncleared
    final_z->backward();
    assert(close_enough((*intermediate_y->grad)[0], 6.0));  // 3.0 + 3.0
    assert(close_enough((*base_x->grad)[0], 24.0));        // 12.0 + 12.0

    // apply explicit zero_grad to clear intermediate accumulation
    intermediate_y->zero_grad();
    base_x->zero_grad();
    assert(close_enough((*intermediate_y->grad)[0], 0.0));
    assert(close_enough((*base_x->grad)[0], 0.0));
    std::cout << "[PASS] explicit zero_grad on intermediate nodes verified successfully\n";

    std::cout << "[PASS] all tests passed cleanly\n";
    return 0;
}