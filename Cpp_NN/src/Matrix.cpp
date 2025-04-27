#include "../include/Matrix.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Matrix
template <typename T>
Matrix<T>::Matrix(int i_rows, int i_cols)
    : rows{i_rows}, cols{i_cols}, table(i_rows, std::vector<T>(i_cols, 0)) {}

template <typename T> Matrix<T>::Matrix(){};

template <typename T> Matrix<T>::Matrix(const Matrix<T> &original) {
  rows = original.rows;
  cols = original.cols;
  table = original.table;
};

template <typename T> void Matrix<T>::populate_data(std::string file_name) {
  std::ifstream file(file_name);
  std::string line;

  if (!file.is_open()) {
    throw std::runtime_error("Could not open file.");
  }

  T val;

  while (std::getline(file, line)) {
    std::stringstream ss(line);

    std::vector<T> row;

    while (ss >> val) {

      row.push_back(val);

      if (ss.peek() == ',') {
        ss.ignore();
      }
    }

    table.push_back(row);
  }

  file.close();
};

template <typename T> std::vector<T> Matrix<T>::get_labels() {
  std::vector<T> result;

  for (int i = 0; i < table.size(); i++) {
    result.push_back(table[i][0]); // implemented in transposed input
  }

  return result;
}

template <typename T> std::vector<T> &Matrix<T>::operator[](int i) {
  return table[i];
}

template <typename T> void Matrix<T>::set_row(int i, std::vector<T> &new_row) {
  table[i] = new_row;
}

template <typename T> void Matrix<T>::transpose() {
  std::vector<std::vector<T>> transposed(table[0].size(),
                                         std::vector<T>(table.size(), 0));

  for (int i = 0; i < table.size(); ++i) {
    for (int j = 0; j < table[0].size(); ++j) {
      transposed[j][i] = table[i][j];
    }
  }
  table = transposed;
}

template <typename T> void Matrix<T>::remove_rows(int &row_pos) {
  for (int i = 0; i < table.size(); ++i) {
    for (int j = 0; j < table[0].size(); ++j) {
      if (j == row_pos) {
        table[i].erase(table[i].begin() + j);
        break;
      }
    }
  }
}

template <typename T> void Matrix<T>::frob_norm(double &max_gradient_norm) {
  double norm = 0.0;
  for (int i = 0; i < table.size(); ++i) {
    for (int j = 0; j < table[0].size(); ++j) {
      norm += table[i][j] * table[i][j]; // Sum of squares
    }
  }

  norm = std::sqrt(norm);

  if (norm > max_gradient_norm) {
    for (int i = 0; i < table.size(); ++i) {
      for (int j = 0; j < table[0].size(); ++j) {
        table[i][j] = table[i][j] / norm * max_gradient_norm; // Sum of squares
      }
    }
  }
}

template <typename T> Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
  if (this != &other) { // Self-assignment check
    rows = other.rows;
    cols = other.cols;
    table = other.table; // Vector handles its own memory management
  }
  return *this; // Return *this to allow for chaining (e.g., a = b = c)
}

template class Matrix<double>;
template class Matrix<int>;
