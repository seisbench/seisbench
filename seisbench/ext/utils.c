#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN /* Make "s#" use Py_ssize_t rather than int. */

#include <Python.h>
#include <float.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

typedef struct {
  const char *data;
  npy_intp sample_stride; // in bytes
  npy_intp channel_stride; // in bytes
} Window;

static PyObject *stack_windows(PyObject *self, PyObject *args, PyObject *kwds) {
  PyObject *offsets_py, *windows_py;
  PyArrayObject *offsets;
  char *method = "avg";

  static char *kwlist[] = {"windows", "offsets", "method", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|s", kwlist, &windows_py,
                                   &offsets_py, &method))
    return NULL;

  if (strcmp(method, "max") != 0 && strcmp(method, "avg") != 0) {
    PyErr_SetString(PyExc_ValueError, "method must be either 'max' or 'avg'");
    return NULL;
  }

  if (!PyArray_Check(offsets_py)) {
    PyErr_SetString(PyExc_ValueError, "offsets is not a NumPy array");
    return NULL;
  }
  offsets = (PyArrayObject *)offsets_py;

  if (!PyArray_IS_C_CONTIGUOUS(offsets)) {
    PyErr_SetString(PyExc_ValueError, "Arrays must be C-contiguous");
    return NULL;
  }
  if (PyArray_TYPE(offsets) != NPY_INTP) {
    PyErr_SetString(PyExc_ValueError, "Offsets array must be of type int");
    return NULL;
  }
  if (PyArray_NDIM(offsets) != 1) {
    PyErr_SetString(PyExc_ValueError, "Offsets array must be 1-dimensional");
    return NULL;
  }

  // Windows are either a 3D array (window, sample, channel) or a sequence of
  // 2D arrays (sample, channel). Both may be strided, e.g., views into a
  // batch of predictions, so no copy is needed to gather them.
  PyObject *windows_seq = NULL; // Keeps the sequence items alive
  Window *windows = NULL;
  PyArrayObject *result = NULL;
  npy_intp n_windows, window_samples, n_channels;

  if (PyArray_Check(windows_py)) {
    PyArrayObject *array = (PyArrayObject *)windows_py;
    if (PyArray_TYPE(array) != NPY_FLOAT32) {
      PyErr_SetString(PyExc_ValueError, "pred_windows must be of type float32");
      return NULL;
    }
    if (PyArray_NDIM(array) != 3) {
      PyErr_SetString(PyExc_ValueError, "Window array must be 3-dimensional");
      return NULL;
    }
    npy_intp *dims = PyArray_DIMS(array);
    npy_intp *strides = PyArray_STRIDES(array);
    n_windows = dims[0];
    window_samples = dims[1];
    n_channels = dims[2];
    windows = (Window *)malloc((n_windows + 1) * sizeof(Window));
    if (windows == NULL)
      return PyErr_NoMemory();
    for (npy_intp i = 0; i < n_windows; i++) {
      windows[i].data = PyArray_BYTES(array) + i * strides[0];
      windows[i].sample_stride = strides[1];
      windows[i].channel_stride = strides[2];
    }
  } else {
    windows_seq = PySequence_Fast(
        windows_py, "windows must be a NumPy array or a sequence of arrays");
    if (windows_seq == NULL)
      return NULL;
    n_windows = PySequence_Fast_GET_SIZE(windows_seq);
    if (n_windows == 0) {
      PyErr_SetString(PyExc_ValueError, "windows must not be empty");
      goto fail;
    }
    windows = (Window *)malloc(n_windows * sizeof(Window));
    if (windows == NULL) {
      PyErr_NoMemory();
      goto fail;
    }
    PyObject **items = PySequence_Fast_ITEMS(windows_seq);
    for (npy_intp i = 0; i < n_windows; i++) {
      if (!PyArray_Check(items[i])) {
        PyErr_SetString(PyExc_ValueError, "windows must be NumPy arrays");
        goto fail;
      }
      PyArrayObject *array = (PyArrayObject *)items[i];
      if (PyArray_TYPE(array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_ValueError, "pred_windows must be of type float32");
        goto fail;
      }
      if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Windows must be 2-dimensional");
        goto fail;
      }
      if (i == 0) {
        window_samples = PyArray_DIM(array, 0);
        n_channels = PyArray_DIM(array, 1);
      } else if (PyArray_DIM(array, 0) != window_samples ||
                 PyArray_DIM(array, 1) != n_channels) {
        PyErr_SetString(PyExc_ValueError, "All windows must have the same shape");
        goto fail;
      }
      windows[i].data = PyArray_BYTES(array);
      windows[i].sample_stride = PyArray_STRIDE(array, 0);
      windows[i].channel_stride = PyArray_STRIDE(array, 1);
    }
  }

  if (PyArray_SIZE(offsets) != n_windows) {
    PyErr_SetString(
        PyExc_ValueError,
        "Offsets array size must match the number of windows in pred_windows");
    goto fail;
  }

  npy_intp *offsets_data = (npy_intp *)PyArray_DATA(offsets);

  // Determine size of output array
  npy_intp size_out = -1;
  for (npy_intp i = 0; i < n_windows; i++) {
    if (offsets_data[i] < 0) {
      PyErr_SetString(PyExc_ValueError, "Offsets must be non-negative");
      goto fail;
    }
    size_out = size_out >= offsets_data[i] ? size_out : offsets_data[i];
  }
  size_out += window_samples;

  // Create the output array
  npy_intp out_dims[2] = {size_out, n_channels};
  result = (PyArrayObject *)PyArray_ZEROS(2, out_dims, NPY_FLOAT32, 0);
  if (result == NULL)
    goto fail;

  float *result_data = (float *)PyArray_DATA(result);
  npy_intp i_win, i_sample, i_channel, offset_res;
  int no_memory = 0;

  Py_BEGIN_ALLOW_THREADS;
  if (!strcmp(method, "max")) {
    for (npy_intp i = 0; i < size_out * n_channels; i++) {
      result_data[i] = NAN;
    }
    for (i_win = 0; i_win < n_windows; i_win++) {
      Window win = windows[i_win];
      for (i_sample = 0; i_sample < window_samples; i_sample++) {
        offset_res = (offsets_data[i_win] + i_sample) * n_channels;
        const char *sample = win.data + i_sample * win.sample_stride;
        for (i_channel = 0; i_channel < n_channels; i_channel++) {
          float value = *(const float *)(sample + i_channel * win.channel_stride);
          // Maximum value arrangement
          result_data[offset_res + i_channel] =
              fmaxf(value, result_data[offset_res + i_channel]);
        }
      }
    }
  } else if (!strcmp(method, "avg")) {
    // A sample is covered by at most n_windows windows, so int32 counts suffice
    npy_int32 *sample_count =
        (npy_int32 *)calloc(size_out * n_channels, sizeof(npy_int32));
    if (sample_count == NULL) {
      no_memory = 1;
    } else {
      for (i_win = 0; i_win < n_windows; i_win++) {
        Window win = windows[i_win];
        float *window_result = result_data + offsets_data[i_win] * n_channels;
        npy_int32 *window_count = sample_count + offsets_data[i_win] * n_channels;
        // NaN values are added as 0 and not counted. Branch-free, so the
        // compiler can vectorize the loops.
        if (win.channel_stride == sizeof(float) &&
            win.sample_stride == n_channels * (npy_intp)sizeof(float)) {
          // Contiguous window: a single flat loop
          const float *values = (const float *)win.data;
          for (npy_intp i = 0; i < window_samples * n_channels; i++) {
            int valid = !isnan(values[i]);
            window_result[i] += valid ? values[i] : 0.0f;
            window_count[i] += valid;
          }
        } else {
          for (i_channel = 0; i_channel < n_channels; i_channel++) {
            const char *channel = win.data + i_channel * win.channel_stride;
            for (i_sample = 0; i_sample < window_samples; i_sample++) {
              float value =
                  *(const float *)(channel + i_sample * win.sample_stride);
              int valid = !isnan(value);
              window_result[i_sample * n_channels + i_channel] +=
                  valid ? value : 0.0f;
              window_count[i_sample * n_channels + i_channel] += valid;
            }
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
  }
  Py_END_ALLOW_THREADS;

  free(windows);
  Py_XDECREF(windows_seq);
  if (no_memory) {
    Py_DECREF(result);
    return PyErr_NoMemory();
  }
  return (PyObject *)result;

fail:
  free(windows);
  Py_XDECREF(windows_seq);
  Py_XDECREF(result);
  return NULL;
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
