#include "Layer.h"
#include "Matrix.h"
#include <iostream>
#include <string>

class Neural_Network {
public:
  Neural_Network(Matrix<double> x, Matrix<double> y, int n_of_layers, int iter,
                 double alpha);

  std::vector<Layer> create_layers();
  std::vector<Matrix<double>> gradient_decent();
  Matrix<double> get_predictions();
  double get_accuracy(Matrix<double> &predictions);
  std::vector<Matrix<double>> weights;
  std::vector<Matrix<double>> biases;
  std::vector<Layer> layers;
  Matrix<double> x;
  Matrix<double> y;

private:
  int n_of_layers;
  int iter;
  double alpha;
};
