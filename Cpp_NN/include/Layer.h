#ifndef LAYER_H
#define LAYER_H
#include "Matrix.h"
#include "Matrix_operations.h"
#include <iostream>
#include <string>
#include <vector>

class Layer {
public:
  Layer(Matrix<double> weights, Matrix<double> biases, double alpha,
        bool is_last, bool is_first);

  std::vector<Matrix<double>> forward_prop(Matrix<double> &A);
  Matrix<double> activation_function(bool is_last, Matrix<double> &z);
  std::vector<Matrix<double>> back_prop(Matrix<double> &a1, Matrix<double> &a2,
                                        Matrix<double> &weights,
                                        Matrix<double> &z1,
                                        Matrix<double> &da_prev,
                                        Matrix<double> &x, Matrix<double> &y);
  std::vector<Matrix<double>> update_params(Matrix<double> &dw,
                                            Matrix<double> &db, double alpha);

  bool is_last;
  bool is_first;
  std::vector<Matrix<double>> R;
  Matrix<double> weights;
  Matrix<double> biases;

private:
  double alpha;
};
#endif
