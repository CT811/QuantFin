#include "../include/Layer.h"
#include "../include/Matrix.h"
#include "../include/Matrix_operations.h"
#include <vector>

Layer::Layer(Matrix<double> i_weights, Matrix<double> i_biases, double i_alpha,
             bool i_is_last, bool i_is_first)
    : weights{i_weights}, biases{i_biases}, is_last{i_is_last}, alpha{i_alpha},
      is_first{i_is_first} {};

Matrix<double> Layer::activation_function(bool is_last, Matrix<double> &z) {
  int row_to_iter = z.get_rows();
  int cols_to_iter = z.get_cols();
  Matrix<double> result(row_to_iter, cols_to_iter);

  // Last layer has soft_max as activation function.
  if (is_last) { // soft_max activation function
    for (int j = 0; j < cols_to_iter; ++j) {
      std::vector<double> output_layer(row_to_iter);
      for (int i = 0; i < row_to_iter; ++i) {
        output_layer[i] = z[i][j];
      }
      std::vector<double> res_soft_max = soft_max_stable(output_layer);
      for (int i = 0; i < row_to_iter; ++i) {
        result[i][j] = res_soft_max[i];
      }
    }
  } else { // ReLu activation function.
    for (int i = 0; i < row_to_iter; ++i) {
      for (int j = 0; j < cols_to_iter; ++j) {
        if (z[i][j] > 0) {
          result[i][j] = z[i][j];
        } else {
          result[i][j] = 0.01 * z[i][j];
        }
      }
    }
  }
  return result;
}

std::vector<Matrix<double>> Layer::forward_prop(Matrix<double> &A) {
  Matrix<double> z1;
  Matrix<double> z2;
  Matrix<double> a;
  char add_sign = '+';
  std::vector<Matrix<double>> R(2);

  z1 = matrix_multiplication(weights, A);
  z2 = matrix_addition(z1, biases, add_sign);
  a = activation_function(is_last, z2);

  R[0] = z2;
  R[1] = a;

  return R;
}

std::vector<Matrix<double>>
Layer::back_prop(Matrix<double> &a1, Matrix<double> &a2,
                 Matrix<double> &weights, Matrix<double> &z1,
                 Matrix<double> &da_prev, Matrix<double> &x,
                 Matrix<double> &y) {
  std::vector<Matrix<double>> result(4);
  double m = y.get_rows();
  Matrix<double> one_hot_y;
  char minus_sign = '-';
  char div_sign = '/';
  Matrix<double> dz;
  Matrix<double> da;
  Matrix<double> dw;
  Matrix<double> db;
  Matrix<double> relu_deriv(z1.get_rows(), z1.get_cols());
  std::vector<double> y_copy = y.get_labels();

  if (is_last) {
    one_hot_y = one_hot_array(y_copy);
    dz = matrix_addition(a2, one_hot_y, minus_sign);

    // Calculate and print loss
    double mean_loss = calculate_loss(a2, one_hot_y);
    std::cout << mean_loss << std::endl;

  } else {

    // if (is_first) {
    //   weights.transpose();
    // }

    // Derivative of ReLu actication function.
    for (int i = 0; i < z1.get_rows(); ++i) {
      for (int j = 0; j < z1.get_cols(); ++j) {
        if (z1[i][j] > 0) {
          relu_deriv[i][j] = 1;
        } else {
          relu_deriv[i][j] = 0;
        }
      }
    }
    dz = element_wise_mult(da_prev, relu_deriv);
  }

  a1.transpose();

  if (is_first) {
    // We want to transpose only for the calculation.
    dw = matrix_multiplication(dz, x);
  } else {
    dw = matrix_multiplication(dz, a1);
  }

  db = sum_cols(dz);

  if (!is_first) {
    da = matrix_multiplication(weights, dz);
  }

  result[0] = dw;
  result[1] = db;
  result[2] = dz;
  result[3] = da;

  return result;
}

std::vector<Matrix<double>>
Layer::update_params(Matrix<double> &dw, Matrix<double> &db, double alpha) {
  std::vector<Matrix<double>> result(2);
  Matrix<double> new_weights;
  Matrix<double> new_biases;
  char mult_sign = '*';
  char min_sign = '-';
  Matrix<double> alpha_weights =
      multiplication_by_constant(dw, alpha, mult_sign);
  new_weights = matrix_addition(weights, alpha_weights, min_sign);
  Matrix<double> alpha_biases =
      multiplication_by_constant(db, alpha, mult_sign);
  new_biases = matrix_addition(biases, alpha_biases, min_sign);
  result[0] = new_weights;
  result[1] = new_biases;

  return result;
}
