#include "../include/Matrix_operations.h"
#include "../include/Matrix.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

Matrix<double> matrix_multiplication(Matrix<double> &a, Matrix<double> &b) {
  int r1 = a.get_rows();
  int c1 = a.get_cols();
  int r2 = b.get_rows();
  int c2 = b.get_cols();
  Matrix<double> mult(r1, c2);

  if (c1 != r2) {
    throw std::runtime_error("Number of columns of the first matrix must be "
                             "the same as the rows of the second.");
  }

  for (int i = 0; i < r1; ++i) {
    for (int j = 0; j < c2; ++j) {
      mult[i][j] = 0;
    }
  }

  for (int i = 0; i < r1; ++i) {
    for (int j = 0; j < c2; ++j) {
      for (int k = 0; k < c1; ++k) {
        mult[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return mult;
};

Matrix<double> matrix_addition(Matrix<double> &a, Matrix<double> &b,
                               char &sign) {
  int r1 = a.get_rows();
  int c1 = a.get_cols();
  int r2 = b.get_rows();
  int c2 = b.get_cols();
  Matrix<double> sum_m(r1, c1);

  if (c2 != 1 | c1 == c2) {
    if (sign == '-') {
      for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
          sum_m[i][j] = a[i][j] - b[i][j];
        }
      }
    } else {
      for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
          sum_m[i][j] = a[i][j] + b[i][j];
        }
      }
    }
  } else if (c2 == 1) {
    if (sign == '-') {
      for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
          sum_m[i][j] = a[i][j] - b[i][0];
        }
      }
    } else {
      for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
          sum_m[i][j] = a[i][j] + b[i][0];
        }
      }
    }
  }

  return sum_m;
};

std::vector<double> soft_max_stable(std::vector<double> &z) {
  std::vector<double> result(z.size());
  double exp_sum = 0.0;
  double max_element = *std::max_element(z.begin(), z.end());

  for (int i = 0; i < z.size(); ++i) {
    exp_sum += std::exp(z[i] - max_element);
  }

  for (int i = 0; i < z.size(); ++i) {
    result[i] = std::exp(z[i] - max_element) / exp_sum;
  }

  return result;
};

Matrix<double> one_hot_array(std::vector<double> &y) {
  Matrix<double> result(10, y.size());

  for (int i = 0; i < y.size(); ++i) {
    std::vector<double> one_h_vec(10, 0);
    one_h_vec[y[i]] = 1;
    for (int k = 0; k < 10; ++k) {
      result[k][i] = one_h_vec[k];
    }
  }

  return result;
};

Matrix<double> multiplication_by_constant(Matrix<double> &a, double &m,
                                          char &sign) {
  if (sign == '/') {
    for (int i = 0; i < a.get_rows(); ++i) {
      for (int j = 0; j < a.get_cols(); ++j) {
        a[i][j] /= m;
      }
    }
  } else {
    for (int i = 0; i < a.get_rows(); ++i) {
      for (int j = 0; j < a.get_cols(); ++j) {
        a[i][j] *= m;
      }
    }
  }

  return a;
};

Matrix<double> sum_cols(Matrix<double> &a) {
  Matrix<double> result(a.get_rows(), 1);
  double sum_c;

  for (int i = 0; i < a.get_rows(); ++i) {
    for (int j = 0; j < a.get_cols(); ++j) {
      sum_c += a[i][j];
    }
    result[i][0] = sum_c;
  }

  return result;
};

Matrix<double> randomize(Matrix<double> &a, double &maximum, double &minimum) {
  int seed = 1337;
  std::srand(seed);

  for (int i = 0; i < a.get_rows(); ++i) {
    for (int j = 0; j < a.get_cols(); ++j) {
      a[i][j] =
          minimum + (static_cast<double>(std::rand()) / (RAND_MAX + 1.0)) *
                        (maximum - minimum);
    }
  }

  return a;
};

std::vector<double> extract_predictions(Matrix<double> &p) {
  int cols_to_iter = p.get_cols();
  std::vector<double> result(cols_to_iter);
  int max_index = 0;

  for (int j = 0; j < cols_to_iter; ++j) {
    double max = p[0][j];
    for (int i = 0; i < p.get_rows(); ++i) {
      if (max < p[i][j]) {
        max = p[i][j];
        max_index = i;
      }
    }

    result[j] = max_index;
  }

  return result;
}

Matrix<double> element_wise_mult(Matrix<double> &a, Matrix<double> &b) {
  int r1 = a.get_rows();
  int c1 = a.get_cols();
  int r2 = b.get_rows();
  int c2 = b.get_cols();
  Matrix<double> mult(r1, c1);

  if (c1 != c2 || r1 != r2) {
    throw std::runtime_error(
        "Dot product needs matrices with the same dimensions.");
  }

  for (int i = 0; i < r1; ++i) {
    for (int j = 0; j < c2; ++j) {
      mult[i][j] = a[i][j] * b[i][j];
    }
  }

  return mult;
}

double calculate_loss(Matrix<double> &y_pred, Matrix<double> &y_actual) {
  double result;
  Matrix<double> log_pred(y_pred.get_rows(), y_pred.get_cols());
  Matrix<double> losses;

  for (int i = 0; i < y_pred.get_rows(); ++i) {
    for (int j = 0; j < y_pred.get_cols(); ++j) {
      log_pred[i][j] = -log(y_pred[i][j] + 1e-8);
    }
  }

  losses = element_wise_mult(y_actual, log_pred);

  double sample_size = losses.get_cols();
  double total_loss = 0;
  for (int j = 0; j < losses.get_cols(); ++j) {
    double loss_i = 0;
    for (int i = 0; i < losses.get_rows(); ++i) {
      loss_i += losses[i][j];
    }
    total_loss += loss_i;
  }

  result = total_loss / sample_size;

  return result;
}
