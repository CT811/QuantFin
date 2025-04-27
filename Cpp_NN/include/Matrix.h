#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <string>

template <typename T> class Matrix {
public:
  class Invalid {};
  std::vector<std::vector<T>> table;
  Matrix(int rows, int cols);
  Matrix();
  // Copy constructor:
  Matrix(const Matrix<T> &original);

  int get_rows() const { return table.size(); };
  int get_cols() const { return table[0].size(); };
  std::vector<T> get_labels();
  void populate_data(std::string file_name);
  std::vector<T> &operator[](int i);
  Matrix<T> &operator=(const Matrix<T> &other);
  void set_row(int i, std::vector<T> &new_row);
  void transpose();
  void remove_rows(int &row_pos);
  void frob_norm(double &max_gradient_norm);
  int rows;
  int cols;
};
#endif
