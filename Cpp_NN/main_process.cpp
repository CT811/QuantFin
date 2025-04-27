#include "include/Layer.h"
#include "include/Matrix.h"
#include "include/Matrix_operations.h"
#include "include/Neural_network.h"
#include <string>
#include <vector>

int main() {

  Matrix<double> x;
  std::vector<double> y_placeholder;

  int label_row = 0;
  x.populate_data("./data/mnist_test.csv");
  // Normalize input to be between 0 and 1 instead of 0 and 255
  char div_sign = '/';
  double max_pixels = 255;
  y_placeholder = x.get_labels();
  Matrix<double> x_copy = multiplication_by_constant(x, max_pixels, div_sign);
  x_copy.remove_rows(label_row);
  int s = y_placeholder.size();
  Matrix<double> y{s, 1};
  for (int i = 0; i < y_placeholder.size(); ++i) {
    y[i][0] = y_placeholder[i];
  }

  int n_of_layers = 3;
  int iterations = 1000;
  double alpha = 0.05;
  double train_accuracy;
  std::vector<Matrix<double>> gd_result(3);
  Matrix<double> predictions;
  Neural_Network my_NN{x_copy, y, n_of_layers, iterations, alpha};
  Layer test = my_NN.layers[0];
  gd_result = my_NN.gradient_decent();
  std::cout << "Done training" << std::endl;
  predictions = my_NN.get_predictions();
  train_accuracy = my_NN.get_accuracy(predictions);
  std::cout << train_accuracy << std::endl;
  return 0;
}
