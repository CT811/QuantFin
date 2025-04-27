#include "../include/Neural_network.h"
#include "../include/Layer.h"
#include "../include/Matrix.h"
#include "../include/Matrix_operations.h"

Neural_Network::Neural_Network(Matrix<double> i_x, Matrix<double> i_y,
                               int i_n_of_layers, int i_iter, double i_alpha)
    : x{i_x}, y{i_y}, n_of_layers{i_n_of_layers}, iter{i_iter}, alpha{i_alpha} {
  // Create random matrices for weights and biases initial parameters
  weights.resize(n_of_layers);
  biases.resize(n_of_layers);
  double min = -0.05;
  double max = 0.05;
  double min_weights = -0.5;
  double max_weights = 0.5;
  Matrix<double> first_layer_weights(10, 784);
  Matrix<double> first_layer_biases(10, 1);
  first_layer_weights =
      randomize(first_layer_weights, max_weights, min_weights);
  first_layer_biases = randomize(first_layer_biases, max, min);

  weights[0] = first_layer_weights;
  biases[0] = first_layer_biases;

  for (int i = 1; i < i_n_of_layers; ++i) {
    Matrix<double> n_layer_weights(10, 10);
    n_layer_weights = randomize(n_layer_weights, max, min);
    weights[i] = n_layer_weights;

    Matrix<double> n_layer_biases(10, 1);
    n_layer_biases = randomize(n_layer_biases, max, min);
    biases[i] = n_layer_biases;
  }

  // Also create the layers vector
  layers = create_layers();
};

std::vector<Layer> Neural_Network::create_layers() {
  if (n_of_layers < 2) {
    throw std::runtime_error("Number of layers must be at least 2.");
  }

  std::vector<Layer> result;

  for (int i = 0; i < n_of_layers; ++i) {
    bool is_first = (i == 0);              // True only for the first layer
    bool is_last = (i == n_of_layers - 1); // True only for the last layer

    Layer current_layer(weights[i], biases[i], alpha, is_last, is_first);
    result.push_back(current_layer); // Add to the end of the vector
  }

  return result;
}

std::vector<Matrix<double>> Neural_Network::gradient_decent() {
  std::vector<Matrix<double>> result(3);
  int n = layers.size();
  for (int i = 0; i < iter; ++i) {
    double progress = ((i + 1.0) / iter) * 100;
    std::cout << "Training the model..." << progress << "%" << std::endl;
    std::vector<Matrix<double>> z(n);
    std::vector<Matrix<double>> a(n);
    std::vector<Matrix<double>> dw(n);
    std::vector<Matrix<double>> db(n);
    std::vector<Matrix<double>> dz(n);
    std::vector<Matrix<double>> da(n);
    std::vector<Matrix<double>> updated_params(2);

    // Forward prop
    for (int j = 0; j < n; ++j) {
      std::vector<Matrix<double>> fw_p;
      if (layers[j].is_first) {
        Matrix<double> x_copy = x;
        x_copy.transpose();
        fw_p = layers[j].forward_prop(x_copy);
        z[0] = fw_p[0];
        a[0] = fw_p[1];
      } else {
        fw_p = layers[j].forward_prop(a[j - 1]);
        z[j] = fw_p[0];
        a[j] = fw_p[1];
      }
    }
    //  Back prop
    for (int j = n - 1; j >= 0; --j) {
      std::vector<Matrix<double>> b_p;
      if (layers[j].is_last) {
        // There is no j+1 dz_prev argument is not used anyway so we pass the
        // same as z[j].
        b_p = layers[j].back_prop(a[j - 1], a[j], layers[j].weights, z[j], z[j],
                                  x, y);
      } else if (layers[j].is_first) {
        b_p = layers[j].back_prop(a[j], a[j], layers[j + 1].weights, z[j],
                                  da[j + 1], x, y);

      } else {
        b_p = layers[j].back_prop(a[j - 1], a[j], layers[j].weights, z[j],
                                  da[j + 1], x, y);
      }
      // Gradient clipping
      Matrix<double> d_w_copy = b_p[0];
      Matrix<double> d_b_copy = b_p[1];
      double max_grad_norm = 1.0;
      d_w_copy.frob_norm(max_grad_norm);
      d_b_copy.frob_norm(max_grad_norm);
      dw[j] = d_w_copy;
      db[j] = d_b_copy;
      dz[j] = b_p[2];
      da[j] = b_p[3];
      // Update params
      updated_params = layers[j].update_params(dw[j], db[j], alpha);
      layers[j].weights = updated_params[0];
      layers[j].biases = updated_params[1];
    }
  }

  result[0] = weights[n - 1];
  result[1] = biases[n - 1];

  return result;
};

Matrix<double> Neural_Network::get_predictions() {
  Matrix<double> result;
  int n = layers.size();
  std::vector<Matrix<double>> z(n);
  std::vector<Matrix<double>> a(n);

  for (int j = 0; j < n; ++j) {
    std::vector<Matrix<double>> fw_p;
    if (layers[j].is_first) {
      Matrix<double> x_copy = x;
      x_copy.transpose();
      fw_p = layers[j].forward_prop(x_copy);
      z[0] = fw_p[0];
      a[0] = fw_p[1];
    } else {
      fw_p = layers[j].forward_prop(a[j - 1]);
      z[j] = fw_p[0];
      a[j] = fw_p[1];
    }
  }

  result = a[n - 1];

  Matrix<double> probs = a[n - 1]; // or whatever holds final softmax output

  return result;
}

double Neural_Network::get_accuracy(Matrix<double> &predictions) {
  std::vector<double> extracted_predictions = extract_predictions(predictions);
  double correct = 0;
  double result;

  for (int i = 0; i < extracted_predictions.size(); ++i) {
    if (extracted_predictions[i] == y[i][0]) {
      correct++;
    }
  }

  result = correct / y.get_rows();

  return result;
};
