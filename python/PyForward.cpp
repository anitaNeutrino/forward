#include "forward/forward.hpp"
#include <complex>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
using namespace forward;

// Numpy real and complex arrays
using NumpyReal    = py::array_t<double>;
using NumpyComplex = py::array_t<std::complex<double>>;

template <typename T>
auto
matrix_to_array(const std::vector<std::vector<T>>& matrix) -> py::array_t<T> {

  // get the number of rows
  const auto N{matrix.size()};

  // and the number of columns in the matrim
  const auto M{matrix[0].size()};

  // create the matrix
  py::array_t<T> array({ N, M });

  // get a mutable reference
  // auto r = array.mutable_unchecked<2>();
  auto r = array.mutable_data();

  // and fill in the matrix
  for (auto i = 0; i < N; ++i) {
    for (auto j = 0; j < M; ++j) {
      r[i, j] = matrix[i][j];
    }
  }

  // and return the matrix
  return array;
}

// convert a std::vector into a py::array
template <typename T>
auto
vector_to_array(const std::vector<T>& vector) -> py::array_t<T> {
  return py::array_t<T>(vector.size(), vector.data());
}

// convert a py::array into an std::vector
template <typename T>
auto
array_to_vector(const py::array_t<T>& array) -> std::vector<T> {

  // get a pointer to the buffer
  const auto buf{array.request()};

  // and create the vector pointing into the buffer
  return std::vector<T>(static_cast<T*>(buf.ptr), static_cast<T*>(buf.ptr) + array.size());
}

// create our Python module
PYBIND11_MODULE(_forward, m) {

  // set a docstring
  m.doc() = "Deconvolve 1D signals using ForWaRD.";

  // the supported wavelet types
  py::enum_<WaveletType>(m, "WaveletType")
      .value("Meyer", WaveletType::Meyer)
      .value("d8", WaveletType::d8)
      .value("d10", WaveletType::d10)
      .value("d12", WaveletType::d12)
      .value("d14", WaveletType::d14)
      .value("d16", WaveletType::d16)
      .value("d18", WaveletType::d18)
      .value("d20", WaveletType::d20);

  // the supported threshold rules
  py::enum_<ThresholdRule>(m, "ThresholdRule")
      .value("Soft", ThresholdRule::Soft)
      .value("Hard", ThresholdRule::Hard);

  m.def("getBasisMatrix",
        [](const NumpyComplex& u, const NumpyComplex& v, const unsigned int p) -> NumpyComplex {

          // get the basis matrix
          const auto matrix{getBasisMatrix(array_to_vector(u),
                                           array_to_vector(v),
                                           p)};

          // and return it as a 2D numpy array
          return matrix_to_array(matrix);

        }, py::arg("u"), py::arg("v"), py::arg("p"),
        "Compute the basis matrix for a pair of wavelet filters.");
  m.def(
      "filt",
      [](const int N, const WaveletType& filterType)
          -> std::tuple<NumpyComplex, NumpyComplex, NumpyComplex, NumpyComplex> {

        // get filters
        const auto filters{filt(N, filterType)};

        // and convert it a tuple of Numpy arrays
        return std::make_tuple(vector_to_array(std::get<0>(filters)),
                               vector_to_array(std::get<1>(filters)),
                               vector_to_array(std::get<2>(filters)),
                               vector_to_array(std::get<3>(filters)));

      },
      py::arg("N"), py::arg("filterType"),
      "Get the wavelet basis filters.");

  m.def(
      "fwt",
      [](const NumpyComplex& z,
         const unsigned int sdim,
         const NumpyComplex& util,
         const NumpyComplex& vtil) -> NumpyComplex {
        // compute the wavelet transform as an std::vector
        const auto wt{
            fwt(array_to_vector(z), sdim, array_to_vector(util), array_to_vector(vtil))};

        // and return it back as a Python array
        return vector_to_array(wt);
      },
      py::arg("z"),
      py::arg("sdim"),
      py::arg("util"),
      py::arg("vtil"),
      "Perform the forward wavelet transform.");
  m.def(
      "ifwt",
      [](const NumpyComplex& z,
         const unsigned int sdim,
         const NumpyComplex& u,
         const NumpyComplex& v) -> NumpyComplex {
        // compute the wavelet transform as an std::vector
        const auto wt{
            ifwt(array_to_vector(z), sdim, array_to_vector(u), array_to_vector(v))};

        // and return it back as a Python array
        return vector_to_array(wt);
      },
      py::arg("z"),
      py::arg("sdim"),
      py::arg("util"),
      py::arg("vtil"),
      "Perform the inverse forward wavelet transform.");
  m.def(
      "deconvolve",
      [](const NumpyReal& signal,
         const NumpyReal& response,
         const unsigned int p,
         const WaveletType type,
         const double noiseSd,
         const NumpyReal& scaling,
         const NumpyReal& rho,
         const ThresholdRule rule) -> NumpyReal {
        // compute the wavelet transform as an std::vector
        const auto deconvolved{deconvolve(array_to_vector(signal),
                                          array_to_vector(response),
                                          p,
                                          type,
                                          noiseSd,
                                          array_to_vector(scaling),
                                          array_to_vector(rho),
                                          rule)};

        // and return it back as a Python array
        return vector_to_array(deconvolved);
      },
      py::arg("signal"),
      py::arg("response"),
      py::arg("p"),
      py::arg("type"),
      py::arg("noiseSd"),
      py::arg("scaling"),
      py::arg("rho"),
      py::arg("rule"),
      "Perform FoRWarD wavelet deconvolution.");
  m.def(
      "get_wavelets",
      [](const NumpyReal& signal,
         const NumpyReal& response,
         const unsigned int p,
         const WaveletType type,
         const double noiseSd,
         const NumpyReal& scaling,
         const NumpyReal& rho,
         const ThresholdRule rule) -> std::tuple<NumpyComplex, NumpyReal> {

        // compute the wavelet transform as an std::vector
        const auto wave_thresholds{get_wavelets(array_to_vector(signal),
                                                array_to_vector(response),
                                                p,
                                                type,
                                                noiseSd,
                                                array_to_vector(scaling),
                                                array_to_vector(rho),
                                                rule)};

        // extract references to the wavelet coefficients
        const auto wavelets{std::get<0>(wave_thresholds)};

        // extract references to the thresholds
        const auto thresholds{std::get<1>(wave_thresholds)};

        // and return it back as a Python array
        return std::make_tuple(vector_to_array(wavelets),
                               vector_to_array(thresholds));
      },
      py::arg("signal"),
      py::arg("response"),
      py::arg("p"),
      py::arg("type"),
      py::arg("noiseSd"),
      py::arg("scaling"),
      py::arg("rho"),
      py::arg("rule"),
      "Return the wavelet coefficients and thresholds");
  m.def(
      "coeff",
      [](const NumpyComplex& w,
         const unsigned int p,
         const unsigned int q) -> NumpyComplex {
        // and return it back as a Python array
        return vector_to_array(coeff(array_to_vector(w), p, q));
      },
      py::arg("w"),
      py::arg("p"),
      py::arg("q"),
      "Return the q-th wavelet coefficient.");
  m.def(
      "coeff",
      [](const NumpyReal& thresholds,
         const unsigned int p,
         const unsigned int q) -> NumpyReal {
        // and return it back as a Python array
        return vector_to_array(coeff(array_to_vector(thresholds), p, q));
      },
      py::arg("thresholds"),
      py::arg("p"),
      py::arg("q"),
      "Return the q-th threshold vector.");

} // PYBIND11-MODULE
