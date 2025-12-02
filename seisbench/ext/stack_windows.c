#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN /* Make "s#" use Py_ssize_t rather than int. */

#include <Python.h>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "numpy/arrayobject.h"

static PyObject* stack_windows(PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject *offsets_py, *pred_windows_py;
  PyArrayObject *offsets, *pred_windows;
  char* method = "avg";
  int n_threads = 4;

  static char* kwlist[] = {"pred_windows", "offsets", "method", "n_threads",
                           NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|si", kwlist,
                                   &pred_windows_py, &offsets_py, &method,
                                   &n_threads))
    return NULL;

  if (n_threads < 0) {
    PyErr_SetString(PyExc_ValueError, "n_threads must be at least 0");
    return NULL;
  }
  if (strcmp(method, "max") == 0 && strcmp(method, "avg") == 0) {
    PyErr_SetString(PyExc_ValueError, "method must be either 'max' or 'avg'");
    return NULL;
  }

  if (!PyArray_Check(offsets_py)) {
    PyErr_SetString(PyExc_ValueError, "offsets is not a NumPy array");
    return NULL;
  }
  if (!PyArray_Check(pred_windows_py)) {
    PyErr_SetString(PyExc_ValueError, "pred_windows is not a NumPy array");
    return NULL;
  }

  // Convert input objects to NumPy arrays if they are not already
  offsets = (PyArrayObject*)offsets_py;
  pred_windows = (PyArrayObject*)pred_windows_py;

  // Check that arrays are C-contiguous
  if (!PyArray_IS_C_CONTIGUOUS(offsets) ||
      !PyArray_IS_C_CONTIGUOUS(pred_windows)) {
    PyErr_SetString(PyExc_ValueError, "Arrays must be C-contiguous");
    return NULL;
  }

  // Check data types
  if (PyArray_TYPE(offsets) != NPY_INTP) {
    PyErr_SetString(PyExc_ValueError, "Offsets array must be of type int");
    return NULL;
  }
  if (PyArray_TYPE(pred_windows) != NPY_FLOAT32) {
    PyErr_SetString(PyExc_ValueError, "Arrays must be of type float64");
    return NULL;
  }

  // Dimension check
  if (PyArray_NDIM(offsets) != 1) {
    PyErr_SetString(PyExc_ValueError, "Offsets array must be 1-dimensional");
    return NULL;
  }

  if (PyArray_NDIM(pred_windows) != 3) {
    PyErr_SetString(PyExc_ValueError,
                    "Pred_windows array must be 3-dimensional");
    return NULL;
  }
  npy_intp* dims_windows = PyArray_DIMS(pred_windows);
  npy_intp n_windows = dims_windows[0];
  npy_intp window_samples = dims_windows[1];
  npy_intp n_channels = dims_windows[2];

  if (PyArray_SIZE(offsets) != n_windows) {
    PyErr_SetString(
        PyExc_ValueError,
        "Offsets array size must match the number of windows in pred_windows");
    return NULL;
  }

  // Get data pointers
  float* data_windows = (float*)PyArray_DATA(pred_windows);
  npy_intp* data_offsets = (npy_intp*)PyArray_DATA(offsets);

  npy_intp size_out = -1;

  for (npy_intp i = 0; i < n_windows; i++) {
    size_out = size_out >= data_offsets[i] ? size_out : data_offsets[i];
  }
  size_out += window_samples;

  // Create the output array
  npy_intp out_dims[2] = {size_out, n_channels};
  PyArrayObject* result =
      (PyArrayObject*)PyArray_ZEROS(2, out_dims, NPY_FLOAT32, 0);

  float* result_data = (float*)PyArray_DATA(result);

  Py_BEGIN_ALLOW_THREADS;
  if (!strcmp(method, "max")) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (npy_intp i_win = 0; i_win < n_windows; i_win++) {
      npy_intp offset = data_offsets[i_win];

      for (npy_intp i_sample = 0; i_sample < window_samples; i_sample++) {
        // #pragma omp simd safelen(3)
        for (npy_intp i_channel = 0; i_channel < n_channels; i_channel++) {
          npy_intp offset_win = i_win * window_samples * n_channels +
                                i_sample * n_channels + i_channel;
          npy_intp offset_res = (offset + i_sample) * n_channels + i_channel;

          float data_value = data_windows[offset_win];
          // Maximum value arrangement
          result_data[offset_res] = data_value > result_data[offset_res]
                                        ? data_value
                                        : result_data[offset_res];
        }
      }
    }

  } else if (!strcmp(method, "avg")) {
    npy_intp* sample_count =
        (npy_intp*)calloc(size_out * n_channels, sizeof(npy_intp));

#ifdef _OPENMP
#pragma omp
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (npy_intp i_win = 0; i_win < n_windows; i_win++) {
      npy_intp offset = data_offsets[i_win];

      for (npy_intp i_channel = 0; i_channel < n_channels; i_channel++) {
        for (npy_intp i_sample = 0; i_sample < window_samples; i_sample++) {
          npy_intp offset_win = i_win * window_samples * n_channels +
                                i_sample * n_channels + i_channel;
          npy_intp offset_res = (offset + i_sample) * n_channels + i_channel;

          float value = data_windows[offset_win];
          if (isnan(value)) {
            continue;
          }
          // Average value arrangement
          result_data[offset_res] +=
              (data_windows[offset_win] - result_data[offset_res]) /
              (++sample_count[(offset + i_sample) * n_channels + i_channel]);
        }
      }
    }
    free(sample_count);
  }
  Py_END_ALLOW_THREADS;

  return (PyObject*)result;
}

/*
 * Method definition table
 */
static PyMethodDef methods[] = {
    {"stack_windows",                             // Function name
     (PyCFunction)(void (*)(void))stack_windows,  // C function pointer
     METH_VARARGS | METH_KEYWORDS,                // Argument parsing method
     "Re-arrange overlapping prediction arrays into a single array"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

/*
 * Module definition structure
 */
static struct PyModuleDef StackWindowsModule = {
    PyModuleDef_HEAD_INIT,
    "stack_windows",                                            // Module name
    "C extension for arranging overlapping prediction arrays",  // Module
                                                                // docstring
    -1,      // Module state size (-1 means global state)
    methods  // Method table
};

/*
 * Module initialization function
 * Must be named PyInit_<module_name>
 */
PyMODINIT_FUNC PyInit_stack_windows(void) {
  import_array();
  // Create the module
  return PyModule_Create(&StackWindowsModule);
}
