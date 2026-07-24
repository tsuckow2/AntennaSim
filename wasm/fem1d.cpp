#include <cmath>
#include <complex>
#include <algorithm>
#include <vector>

namespace {
using Complex = std::complex<double>;
constexpr double kSpeedOfLight = 299792458.0;
constexpr double kPi = 3.14159265358979323846;

bool solve_system(std::vector<Complex>& matrix, std::vector<Complex>& rhs, int size) {
  for (int column = 0; column < size; ++column) {
    int pivot = column;
    double pivot_size = std::abs(matrix[column * size + column]);
    for (int row = column + 1; row < size; ++row) {
      const double candidate = std::abs(matrix[row * size + column]);
      if (candidate > pivot_size) {
        pivot = row;
        pivot_size = candidate;
      }
    }
    if (pivot_size < 1e-14) return false;
    if (pivot != column) {
      for (int entry = column; entry < size; ++entry) {
        std::swap(matrix[column * size + entry], matrix[pivot * size + entry]);
      }
      std::swap(rhs[column], rhs[pivot]);
    }
    for (int row = column + 1; row < size; ++row) {
      const Complex factor = matrix[row * size + column] / matrix[column * size + column];
      for (int entry = column + 1; entry < size; ++entry) {
        matrix[row * size + entry] -= factor * matrix[column * size + entry];
      }
      rhs[row] -= factor * rhs[column];
    }
  }

  for (int row = size - 1; row >= 0; --row) {
    Complex value = rhs[row];
    for (int entry = row + 1; entry < size; ++entry) {
      value -= matrix[row * size + entry] * rhs[entry];
    }
    rhs[row] = value / matrix[row * size + row];
  }
  return true;
}
}  // namespace

extern "C" {

// Output stride: frequency MHz, Z real, Z imag, S11 real, S11 imag, S11 dB.
int fem_solve(double length_m, double start_mhz, double stop_mhz, int steps,
              double mesh_size_m, double reference_impedance_ohms,
              double* output) {
  if (!output || !std::isfinite(length_m) || length_m <= 0.0 ||
      !std::isfinite(start_mhz) || !std::isfinite(stop_mhz) ||
      start_mhz <= 0.0 || stop_mhz < start_mhz || steps < 1 ||
      !std::isfinite(mesh_size_m) || mesh_size_m <= 0.0 ||
      !std::isfinite(reference_impedance_ohms) || reference_impedance_ohms <= 0.0) {
    return -1;
  }

  const int elements = std::max(2, std::min(300, static_cast<int>(std::ceil(length_m / mesh_size_m))));
  const int nodes = elements + 1;
  const double element_length = length_m / elements;
  std::vector<Complex> stiffness(nodes * nodes, Complex(0.0, 0.0));
  std::vector<Complex> mass(nodes * nodes, Complex(0.0, 0.0));
  const Complex local_k[2][2] = {
    {1.0 / element_length, -1.0 / element_length},
    {-1.0 / element_length, 1.0 / element_length},
  };
  const double local_m[2][2] = {
    {2.0 * element_length / 6.0, element_length / 6.0},
    {element_length / 6.0, 2.0 * element_length / 6.0},
  };
  for (int element = 0; element < elements; ++element) {
    for (int row = 0; row < 2; ++row) {
      for (int column = 0; column < 2; ++column) {
        const int global_row = element + row;
        const int global_column = element + column;
        stiffness[global_row * nodes + global_column] += local_k[row][column];
        mass[global_row * nodes + global_column] += local_m[row][column];
      }
    }
  }

  for (int step = 0; step < steps; ++step) {
    const double fraction = steps == 1 ? 0.0 : static_cast<double>(step) / (steps - 1);
    const double frequency_mhz = start_mhz + (stop_mhz - start_mhz) * fraction;
    const double wavenumber = 2.0 * kPi * frequency_mhz * 1e6 / kSpeedOfLight;
    std::vector<Complex> matrix = stiffness;
    std::vector<Complex> voltage(nodes, Complex(0.0, 0.0));
    voltage[0] = 1.0;
    voltage[nodes - 1] = -1.0;
    for (int row = 0; row < nodes; ++row) {
      for (int column = 0; column < nodes; ++column) {
        matrix[row * nodes + column] -= wavenumber * wavenumber * mass[row * nodes + column];
      }
    }
    matrix[0] += Complex(0.0, 1e-6);
    matrix[nodes * nodes - 1] += Complex(0.0, 1e-6);
    if (!solve_system(matrix, voltage, nodes)) return -2;

    const Complex impedance = voltage[0] - voltage[nodes - 1];
    const Complex reflection = (impedance - reference_impedance_ohms) /
                               (impedance + reference_impedance_ohms);
    const double magnitude = std::max(std::abs(reflection), 1e-12);
    double* row = output + step * 6;
    row[0] = frequency_mhz;
    row[1] = impedance.real();
    row[2] = impedance.imag();
    row[3] = reflection.real();
    row[4] = reflection.imag();
    row[5] = 20.0 * std::log10(magnitude);
  }
  return elements;
}

}
