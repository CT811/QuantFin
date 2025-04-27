#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H
#include "Matrix.h"

Matrix<double> matrix_multiplication(Matrix<double> &a, Matrix<double> &b);
Matrix<double> matrix_addition(Matrix<double> &a, Matrix<double> &b,
                               char &sign);
std::vector<double> soft_max_stable(std::vector<double> &z);
Matrix<double> one_hot_array(std::vector<double> &y);
Matrix<double> multiplication_by_constant(Matrix<double> &a, double &m,
                                          char &sign);
Matrix<double> sum_cols(Matrix<double> &a);
Matrix<double> randomize(Matrix<double> &a, double &maximum, double &minimum);
std::vector<double> extract_predictions(Matrix<double> &p);
Matrix<double> element_wise_mult(Matrix<double> &a, Matrix<double> &b);
double calculate_loss(Matrix<double> &y_pred, Matrix<double> &y_actual);
#endif
