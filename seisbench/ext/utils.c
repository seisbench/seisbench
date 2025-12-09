#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN /* Make "s#" use Py_ssize_t rather than int. */

#include <Python.h>
#include <float.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

static PyObject *stack_windows(PyObject *self, PyObject *args, PyObject *kwds) {
  PyObject *offsets_py, *windows_py;
  PyArrayObject *offsets, *windows;
  char *method = "avg";

  static char *kwlist[] = {"windows", "offsets", "method", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|s", kwlist, &windows_py,
                                   &offsets_py, &method))
    return NULL;

  if (strcmp(method, "max") == 0 && strcmp(method, "avg") == 0) {
    PyErr_SetString(PyExc_ValueError, "method must be either 'max' or 'avg'");
    return NULL;
  }

  if (!PyArray_Check(offsets_py)) {
    PyErr_SetString(PyExc_ValueError, "offsets is not a NumPy array");
    return NULL;
  }
  if (!PyArray_Check(windows_py)) {
    PyErr_SetString(PyExc_ValueError, "pred_windows is not a NumPy array");
    return NULL;
  }

  // Convert input objects to NumPy arrays if they are not already
  offsets = (PyArrayObject *)offsets_py;
  windows = (PyArrayObject *)windows_py;

  // Check that arrays are C-contiguous
  if (!PyArray_IS_C_CONTIGUOUS(offsets) || !PyArray_IS_C_CONTIGUOUS(windows)) {
    PyErr_SetString(PyExc_ValueError, "Arrays must be C-contiguous");
    return NULL;
  }

  // Check data types
  if (PyArray_TYPE(offsets) != NPY_INTP) {
    PyErr_SetString(PyExc_ValueError, "Offsets array must be of type int");
    return NULL;
  }
  if (PyArray_TYPE(windows) != NPY_FLOAT32) {
    PyErr_SetString(PyExc_ValueError, "pred_windows must be of type float64");
    return NULL;
  }

  // Dimension check
  if (PyArray_NDIM(offsets) != 1) {
    PyErr_SetString(PyExc_ValueError, "Offsets array must be 1-dimensional");
    return NULL;
  }

  if (PyArray_NDIM(windows) != 3) {
    PyErr_SetString(PyExc_ValueError, "Window array must be 3-dimensional");
    return NULL;
  }
  npy_intp *dims_windows = PyArray_DIMS(windows);
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
  float *windows_data = (float *)PyArray_DATA(windows);
  npy_intp *offsets_data = (npy_intp *)PyArray_DATA(offsets);

  // Determine size of output array
  npy_intp size_out = -1;
  for (npy_intp i = 0; i < n_windows; i++) {
    size_out = size_out >= offsets_data[i] ? size_out : offsets_data[i];
  }
  size_out += window_samples;

  // Create the output array
  npy_intp out_dims[2] = {size_out, n_channels};
  PyArrayObject *result =
      (PyArrayObject *)PyArray_ZEROS(2, out_dims, NPY_FLOAT32, 0);

  float *result_data = (float *)PyArray_DATA(result);

  npy_intp i_win, i_sample, i_channel;
  npy_intp offset, offset_sample, offset_res, offset_input;

  Py_BEGIN_ALLOW_THREADS;
  if (!strcmp(method, "max")) {
    for (npy_intp i = 0; i < size_out * n_channels; i++) {
      result_data[i] = NAN;
    }
    for (i_win = 0; i_win < n_windows; i_win++) {
      offset = offsets_data[i_win];
      offset_sample = i_win * window_samples * n_channels;

      for (i_sample = 0; i_sample < window_samples; i_sample++) {
        offset_res = (offset + i_sample) * n_channels;
        offset_input = offset_sample + i_sample * n_channels;

        for (i_channel = 0; i_channel < n_channels; i_channel++) {
          // Maximum value arrangement
          result_data[offset_res + i_channel] =
              fmaxf(windows_data[offset_input + i_channel],
                    result_data[offset_res + i_channel]);
        }
      }
    }

  } else if (!strcmp(method, "avg")) {
    npy_intp *sample_count =
        (npy_intp *)calloc(size_out * n_channels, sizeof(npy_intp));
    for (i_win = 0; i_win < n_windows; i_win++) {
      offset = offsets_data[i_win];
      offset_sample = i_win * window_samples * n_channels;

      for (i_sample = 0; i_sample < window_samples; i_sample++) {
        offset_res = (offset + i_sample) * n_channels;
        offset_input = offset_sample + i_sample * n_channels;

        for (i_channel = 0; i_channel < n_channels; i_channel++) {
          if (isnan(windows_data[offset_input + i_channel])) {
            continue;
          }
          // Maximum value arrangement
          result_data[offset_res + i_channel] +=
              windows_data[offset_input + i_channel];
          sample_count[offset_res + i_channel]++;
        }
      }
    }
    for (i_sample = 0; i_sample < size_out * n_channels; i_sample++) {
      if (sample_count[i_sample] > 0) {
        result_data[i_sample] /= sample_count[i_sample];
      } else {
        result_data[i_sample] = NAN;
      }
    }
    free(sample_count);
  }
  Py_END_ALLOW_THREADS;

  return (PyObject *)result;
}

static PyObject *get_edge_indices(PyObject *self, PyObject *args,
                                  PyObject *kwds) {
  PyObject *array_py;
  PyArrayObject *array;
  float edge_value = 0.0;

  static char *kwlist[] = {"array", "edge_value", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|f", kwlist, &array_py,
                                   &edge_value))
    return NULL;

  if (!PyArray_Check(array_py)) {
    PyErr_SetString(PyExc_ValueError, "array is not a NumPy array");
    return NULL;
  }

  // Convert input objects to NumPy arrays if they are not already
  array = (PyArrayObject *)array_py;

  // Check that arrays are C-contiguous
  if (!PyArray_IS_C_CONTIGUOUS(array)) {
    PyErr_SetString(PyExc_ValueError, "array must be C-contiguous");
    return NULL;
  }

  // Check data types
  if (PyArray_TYPE(array) != NPY_FLOAT32) {
    PyErr_SetString(PyExc_ValueError, "array must be of type float32");
    return NULL;
  }

  // Dimension check
  if (PyArray_NDIM(array) != 1) {
    PyErr_SetString(PyExc_ValueError, "array must be 1-dimensional");
    return NULL;
  }

  npy_intp n_samples = PyArray_SIZE(array);
  float *array_data = (float *)PyArray_DATA(array);

  npy_intp begin_edge = 0, end_edge = 0;

  if (isnan(edge_value)) {
    for (npy_intp i = 0; i < n_samples; i++) {
      if (!npy_isnan(array_data[i])) {
        begin_edge = i;
        break;
      }
    }
    for (npy_intp i = n_samples - 1; i >= 0; i--) {
      if (!npy_isnan(array_data[i])) {
        end_edge = i + 1;
        break;
      }
    }
  } else {
    for (npy_intp i = 0; i < n_samples; i++) {
      if (array_data[i] != edge_value) {
        begin_edge = i;
        break;
      }
    }
    for (npy_intp i = n_samples - 1; i >= 0; i--) {
      if (array_data[i] != edge_value) {
        end_edge = i + 1;
        break;
      }
    }
  }
  return Py_BuildValue("(ll)", begin_edge, end_edge);
}

/*
 * Method definition table
 */
static PyMethodDef methods[] = {
    {"stack_windows",                            // Function name
     (PyCFunction)(void (*)(void))stack_windows, // C function pointer
     METH_VARARGS | METH_KEYWORDS,               // Argument parsing method
     "Re-arrange overlapping prediction arrays into a single array"},
    {"get_edge_indices",                            // Function name
     (PyCFunction)(void (*)(void))get_edge_indices, // C function pointer
     METH_VARARGS | METH_KEYWORDS,                  // Argument parsing method
     "Get the number of edge samples with a given value in a 1D array"},
    {NULL, NULL, 0, NULL} // Sentinel
};

/*
 * Module definition structure
 */
static struct PyModuleDef UtilsModule = {
    PyModuleDef_HEAD_INIT,
    "utils", // Module name
    "Utility functions for post-processing SeisBench predictions", // Module
                                                                   // docstring
    -1,     // Module state size (-1 means global state)
    methods // Method table
};

/*
 * Module initialization function
 * Must be named PyInit_<module_name>
 */
PyMODINIT_FUNC PyInit_utils(void) {
  import_array();
  // Create the module
  return PyModule_Create(&UtilsModule);
}
