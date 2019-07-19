
#define _VERSION_ "2018.08.13"

#include <Python.h>
//#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

void sincos(double x, double *sn, double *cs)
{
  *sn = sin(x);
  *cs = cos(x);
}

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

static PyObject *update_estimate_poisson(PyObject *self, PyObject *args)
{
  PyObject* a = NULL;
  PyObject* b = NULL;
  npy_intp sz = 0, i;
  double tmp, tmp2;
  npy_float32* a_data_sp = NULL;
  npy_float32* b_data_sp = NULL;
  npy_complex64* b_data_csp = NULL;
  npy_float64* a_data_dp = NULL;
  npy_float64* b_data_dp = NULL;
  npy_complex128* b_data_cdp = NULL;
  double c, c0, c1, c2;
  double unstable = 0.0, stable = 0.0, negative = 0.0, exact = 0.0;
  if (!PyArg_ParseTuple(args, "OOd", &a, &b, &c))
    return NULL;
  if (c<0 || c>0.5)
    {
      PyErr_SetString(PyExc_TypeError,"third argument must be non-negative and less than 0.5");
      return NULL;
    }
  if (!(PyArray_Check(a) && PyArray_Check(b)))
    {
      PyErr_SetString(PyExc_TypeError,"first two arguments must be array objects");
      return NULL;
    }
  sz = PyArray_SIZE(a);
  if (sz != PyArray_SIZE(b))
    {
      PyErr_SetString(PyExc_TypeError,"array argument sizes must be equal");
      return NULL;
    }
  c0 = -c;
  c1 = 1.0+c;
  c2 = 1.0-c;
  if ((PyArray_TYPE(a) == PyArray_FLOAT32) && (PyArray_TYPE(b) == PyArray_FLOAT32))
    {
      a_data_sp = (npy_float32*)PyArray_DATA(a);
      b_data_sp = (npy_float32*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp = b_data_sp[i];
	  tmp2 = (a_data_sp[i] *= (tmp>0?tmp:0.0));
	  if (tmp==0.0 || tmp==1.0)
	    exact += tmp2;
	  else if (((tmp>c0) && (tmp<c)) || ((tmp<c1) && (tmp>c2)))
	    stable += tmp2;
	  else
	    unstable += tmp2;
	  if (tmp2<0)
	    negative += tmp2;
	}
    }
  else if ((PyArray_TYPE(a) == PyArray_FLOAT32) && (PyArray_TYPE(b) == PyArray_COMPLEX64))
    {
      a_data_sp = (npy_float32*)PyArray_DATA(a);
      b_data_csp = (npy_complex64*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp = b_data_csp[i].real;
	  tmp2 = (a_data_sp[i] *= (tmp>0?tmp:0.0));
	  if (tmp==0.0 || tmp==1.0)
	    exact += tmp2;
	  else if (((tmp>c0) && (tmp<c)) || ((tmp<c1) && (tmp>c2)))
	    stable += tmp2;
	  else
	    unstable += tmp2;
	  if (tmp2<0)
	    negative += tmp2;
	}
    }
  else if ((PyArray_TYPE(a) == PyArray_FLOAT64) && (PyArray_TYPE(b) == PyArray_FLOAT64))
    {
      a_data_dp = (npy_float64*)PyArray_DATA(a);
      b_data_dp = (npy_float64*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp = b_data_dp[i];
	  tmp2 = (a_data_dp[i] *= (tmp>0?tmp:0.0));
	  if (tmp==0.0 || tmp==1.0)
	    exact += tmp2;
	  else if (((tmp>c0) && (tmp<c)) || ((tmp<c1) && (tmp>c2)))
	    stable += tmp2;
	  else
	    unstable += tmp2;
	  if (tmp2<0)
	    negative += tmp2;
	}
    }
  else if ((PyArray_TYPE(a) == PyArray_FLOAT64) && (PyArray_TYPE(b) == PyArray_COMPLEX128))
    {
      a_data_dp = (npy_float64*)PyArray_DATA(a);
      b_data_cdp = (npy_complex128*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp = (b_data_cdp[i]).real;
	  tmp2 = (a_data_dp[i] *= (tmp>0?tmp:0.0));
	  if (tmp==0.0 || tmp==1.0)
	    exact += tmp2;
	  else if (((tmp>c0) && (tmp<c)) || ((tmp<c1) && (tmp>c2)))
	    stable += tmp2;
	  else
	    unstable += tmp2;
	  if (tmp2<0)
	    negative += tmp2;
	}
    }
  else
    {
      PyErr_SetString(PyExc_TypeError,"array argument types must be either float32 or float64");
      return NULL;
    }
  return Py_BuildValue("dddd", exact, stable, unstable, negative);
}

static PyObject *update_estimate_gauss(PyObject *self, PyObject *args)
{
  PyObject* a = NULL;
  PyObject* b = NULL;
  npy_intp sz = 0, i;
  double tmp, tmp2;
  npy_float32* a_data_sp = NULL;
  npy_float32* b_data_sp = NULL;
  npy_complex64* b_data_csp = NULL;
  npy_float64* a_data_dp = NULL;
  npy_float64* b_data_dp = NULL;
  npy_complex128* b_data_cdp = NULL;
  double c, c0, c1, c2, alpha;
  double unstable = 0.0, stable = 0.0, negative=0.0, exact=0.0;
  if (!PyArg_ParseTuple(args, "OOdd", &a, &b, &c, &alpha))
    return NULL;
  if (c<0 || c>0.5)
    {
      PyErr_SetString(PyExc_TypeError,"third argument must be non-negative and less than 0.5");
      return NULL;
    }
  if (!(PyArray_Check(a) && PyArray_Check(b)))
    {
      PyErr_SetString(PyExc_TypeError,"first two arguments must be array objects");
      return NULL;
    }
  sz = PyArray_SIZE(a);
  if (sz != PyArray_SIZE(b))
    {
      PyErr_SetString(PyExc_TypeError,"array argument sizes must be equal");
      return NULL;
    }
  c0 = -c;
  c1 = 1.0+c;
  c2 = 1.0-c;
  if ((PyArray_TYPE(a) == PyArray_FLOAT32) && (PyArray_TYPE(b) == PyArray_FLOAT32))
    {
      a_data_sp = (npy_float32*)PyArray_DATA(a);
      b_data_sp = (npy_float32*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp = a_data_sp[i];
	  tmp2 = (a_data_sp[i] += alpha * b_data_sp[i]);
	  if (tmp==0.0)
	    tmp = 2.0; // force unstable
	  else
	    tmp = a_data_sp[i] / tmp;
	  if (tmp==0.0 || tmp==1.0)
	    exact += tmp2;
	  else if (((tmp>c0) && (tmp<c)) || ((tmp<c1) && (tmp>c2)))
	    stable += tmp2;
	  else
	    unstable += tmp2;
	  if (tmp2<0)
	    negative += tmp2;
	}
    }
  else if ((PyArray_TYPE(a) == PyArray_FLOAT32) && (PyArray_TYPE(b) == PyArray_COMPLEX64))
    {
      a_data_sp = (npy_float32*)PyArray_DATA(a);
      b_data_csp = (npy_complex64*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp = a_data_sp[i];
	  tmp2 = (a_data_sp[i] += alpha * b_data_csp[i].real);
	  if (tmp==0.0)
	    tmp = 2.0; // force unstable
	  else
	    tmp = a_data_sp[i] / tmp;
	  if (tmp==0.0 || tmp==1.0)
	    exact += tmp2;
	  else if (((tmp>c0) && (tmp<c)) || ((tmp<c1) && (tmp>c2)))
	    stable += tmp2;
	  else
	    unstable += tmp2;
	  if (tmp2<0)
	    negative += tmp2;
	}
    }
  else if ((PyArray_TYPE(a) == PyArray_FLOAT64) && (PyArray_TYPE(b) == PyArray_FLOAT64))
    {
      a_data_dp = (npy_float64*)PyArray_DATA(a);
      b_data_dp = (npy_float64*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp = a_data_dp[i];
	  tmp2 = (a_data_dp[i] += alpha * b_data_dp[i]);
	  if (tmp==0.0)
	    tmp = 2.0; // force unstable
	  else
	    tmp = a_data_dp[i] / tmp;
	  if (tmp==0.0 || tmp==1.0)
	    exact += tmp2;
	  else if (((tmp>c0) && (tmp<c)) || ((tmp<c1) && (tmp>c2)))
	    stable += tmp2;
	  else
	    unstable += tmp2;
	  if (tmp2<0)
	    negative += tmp2;
	}
    }
  else if ((PyArray_TYPE(a) == PyArray_FLOAT64) && (PyArray_TYPE(b) == PyArray_COMPLEX128))
    {
      a_data_dp = (npy_float64*)PyArray_DATA(a);
      b_data_cdp = (npy_complex128*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp = a_data_dp[i];
	  tmp2 = (a_data_dp[i] += alpha * b_data_cdp[i].real);
	  if (tmp==0.0)
	    tmp = 2.0; // force unstable
	  else
	    tmp = a_data_dp[i] / tmp;
	  if (tmp==0.0 || tmp==1.0)
	    exact += tmp2;
	  else if (((tmp>c0) && (tmp<c)) || ((tmp<c1) && (tmp>c2)))
	    stable += tmp2;
	  else
	    unstable += tmp2;
	  if (tmp2<0)
	    negative += tmp2;
	}
    }
  else
    {
      PyErr_SetString(PyExc_TypeError,"array argument types must be either float32 or float64");
      return NULL;
    }
  return Py_BuildValue("dddd", exact, stable, unstable, negative);
}


static PyObject *poisson_hist_factor_estimate(PyObject *self, PyObject *args)
{
  PyObject* a = NULL;
  PyObject* b = NULL;
  npy_intp sz = 0, i;
  npy_float32 tmp;
  npy_float32* a_data = NULL;
  npy_float32* b_data = NULL;
  double c;
  double unstable = 0.0, stable = 0.0;
  if (!PyArg_ParseTuple(args, "OOd", &a, &b, &c))
    return NULL;
  if (c<0 || c>0.5)
    {
      PyErr_SetString(PyExc_TypeError,"third argument must be non-negative and less than 0.5");
      return NULL;
    }
  if (!(PyArray_Check(a) && PyArray_Check(b)))
    {
      PyErr_SetString(PyExc_TypeError,"first two arguments must be array objects");
      return NULL;
    }
  sz = PyArray_SIZE(a);
  if (sz != PyArray_SIZE(b))
    {
      PyErr_SetString(PyExc_TypeError,"array argument sizes must be equal");
      return NULL;
    }
  if (! ((PyArray_TYPE(a) == PyArray_FLOAT32) && (PyArray_TYPE(b) == PyArray_FLOAT32)))
    {
      PyErr_SetString(PyExc_TypeError,"array argument types must be float32");
      return NULL;
    }
  a_data = (npy_float32*)PyArray_DATA(a);
  b_data = (npy_float32*)PyArray_DATA(b);
  for (i=0; i<sz; ++i)
    {
      tmp = a_data[i];
      if (((tmp>-c) && (tmp<c)) || ((tmp<1.0+c) && (tmp>1.0-c)))
	stable += tmp * b_data[i];
      else
	{
	  unstable += tmp * b_data[i];
	}
    }
  return Py_BuildValue("dd",stable, unstable);
}

// kldiv(f,f0)=E(f0-f+f*log(f/f0)) f0,f>=level
static PyObject* kullback_leibler_divergence(PyObject *self, PyObject *args)
{
  PyObject* a = NULL;
  PyObject* b = NULL;
  npy_float64 f,f0, level = 1.0;
  npy_intp sz = 0, i, count=0;
  if (!PyArg_ParseTuple(args, "OO|f", &a, &b, &level))
    return NULL; 
  if (!(PyArray_Check(a) && PyArray_Check(b)))
    {
      PyErr_SetString(PyExc_TypeError,"arguments must be array objects");
      return NULL;
    }
  sz = PyArray_SIZE(a);

  if (sz != PyArray_SIZE(b))
    {
      PyErr_SetString(PyExc_TypeError,"argument sizes must be equal");
      return NULL;
    }
  if (PyArray_TYPE(a) != PyArray_TYPE(b))
    {
      PyErr_SetString(PyExc_TypeError,"argument types must be same");
      return NULL;
    }
  level = (level<0? 0.0 : level);
  switch(PyArray_TYPE(a))
    {
    case PyArray_FLOAT64:
      {
	npy_float64 result=0.0;
	for (i=0; i<sz; ++i)
	  {
	    f = *((npy_float64*)PyArray_DATA(a) + i);
	    f0 = *((npy_float64*)PyArray_DATA(b) + i);
	    if (f0<=level || f<level)
	      continue;
	    if (f==0.0)
	      result += f0;
	    else
	      result += f0 - f + f*log(f/f0);
	    count ++;
	  }
	return Py_BuildValue("f", result/count);
      }
      break;
    case PyArray_FLOAT32:
      {
	npy_float64 result=0.0;
	for (i=0; i<sz; ++i)
	  {
	    f = *((npy_float32*)PyArray_DATA(a) + i);
	    f0 = *((npy_float32*)PyArray_DATA(b) + i);
	    if (f0<=level || f<level)
	      continue;
	    if (f==0.0)
	      result += f0;
	    else
	      result += f0 - f + f*log(f/f0);
	    count ++;
	  }
	return Py_BuildValue("f", result/count);
      }
      break;
    default:
      PyErr_SetString(PyExc_TypeError,"argument types must be float64");
      return NULL;
    }
}

static PyObject *zero_if_zero_inplace(PyObject *self, PyObject *args)
{
  PyObject* a = NULL;
  PyObject* b = NULL;
  npy_intp sz = 0, i;
  npy_complex64* tmp = NULL;
  npy_complex64* a_data = NULL;
  npy_float32* b_data = NULL;
  if (!PyArg_ParseTuple(args, "OO", &a, &b))
    return NULL;
  if (!(PyArray_Check(a) && PyArray_Check(b)))
    {
      PyErr_SetString(PyExc_TypeError,"arguments must be array objects");
      return NULL;
    }
  sz = PyArray_SIZE(a);

  if (sz != PyArray_SIZE(b))
    {
      PyErr_SetString(PyExc_TypeError,"argument sizes must be equal");
      return NULL;
    }
  if (! ((PyArray_TYPE(a) == PyArray_COMPLEX64) && (PyArray_TYPE(b) == PyArray_FLOAT32)))
    {
      PyErr_SetString(PyExc_TypeError,"argument types must be complex64 and float32");
      return NULL;
    }
  a_data = (npy_complex64*)PyArray_DATA(a);
  b_data = (npy_float32*)PyArray_DATA(b);
  for (i=0; i<sz; ++i)
    {
      if (b_data[i]==0)
	{
	  tmp = a_data + i;
	  tmp->real = tmp->imag = 0.0;
	}
    }
  return Py_BuildValue("");
}

static PyObject *inverse_division_inplace(PyObject *self, PyObject *args)
{
  PyObject* a = NULL;
  PyObject* b = NULL;
  npy_intp sz = 0, i;
  npy_complex64* tmp_sp = NULL;
  npy_float32 tmp2_sp;
  npy_complex64* a_data_sp = NULL;
  npy_float32* b_data_sp = NULL;
  npy_complex128* tmp_dp = NULL;
  npy_float64 tmp2_dp;
  npy_complex128* a_data_dp = NULL;
  npy_float64* b_data_dp = NULL;
  if (!PyArg_ParseTuple(args, "OO", &a, &b))
    return NULL;
  if (!(PyArray_Check(a) && PyArray_Check(b)))
    {
      PyErr_SetString(PyExc_TypeError,"arguments must be array objects");
      return NULL;
    }
  sz = PyArray_SIZE(a);

  if (sz != PyArray_SIZE(b))
    {
      PyErr_SetString(PyExc_TypeError,"argument sizes must be equal");
      return NULL;
    }
  if ((PyArray_TYPE(a) == PyArray_COMPLEX64) && (PyArray_TYPE(b) == PyArray_FLOAT32))
    {
      a_data_sp = (npy_complex64*)PyArray_DATA(a);
      b_data_sp = (npy_float32*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp_sp = a_data_sp + i;
	  if (tmp_sp->real==0.0 || (b_data_sp[i]==0.0))
	    {
	      tmp_sp->real = tmp_sp->imag = 0.0;
	    }
	  else
	    {
	      tmp2_sp = b_data_sp[i] / (tmp_sp->real * tmp_sp->real + tmp_sp->imag * tmp_sp->imag);
	      tmp_sp->real *= tmp2_sp;
	      tmp_sp->imag *= -tmp2_sp;
	    }
	}
    }
  else if ((PyArray_TYPE(a) == PyArray_COMPLEX128) && (PyArray_TYPE(b) == PyArray_FLOAT64))
    {
      a_data_dp = (npy_complex128*)PyArray_DATA(a);
      b_data_dp = (npy_float64*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp_dp = a_data_dp + i;
	  if (tmp_dp->real==0.0 || (b_data_dp[i]==0.0))
	    {
	      tmp_dp->real = tmp_dp->imag = 0.0;
	    }
	  else
	    {
	      tmp2_dp = b_data_dp[i] / (tmp_dp->real * tmp_dp->real + tmp_dp->imag * tmp_dp->imag);
	      tmp_dp->real *= tmp2_dp;
	      tmp_dp->imag *= -tmp2_dp;
	    }
	}
    }
  else
    {
      PyErr_SetString(PyExc_TypeError,"argument types must be complex64 and float32");
      return NULL;
    }
  return Py_BuildValue("");
}

static PyObject *inverse_subtraction_inplace(PyObject *self, PyObject *args)
{
  PyObject* a = NULL;
  PyObject* b = NULL;
  npy_intp sz = 0, i;
  npy_complex64* tmp_sp = NULL;
  npy_complex64* a_data_sp = NULL;
  npy_float32* b_data_sp = NULL;
  npy_complex128* tmp_dp = NULL;
  npy_complex128* a_data_dp = NULL;
  npy_float64* b_data_dp = NULL;
  double c;
  if (!PyArg_ParseTuple(args, "OOd", &a, &b, &c))
    return NULL;
  if (!(PyArray_Check(a) && PyArray_Check(b)))
    {
      PyErr_SetString(PyExc_TypeError,"arguments must be array objects");
      return NULL;
    }
  sz = PyArray_SIZE(a);

  if (sz != PyArray_SIZE(b))
    {
      PyErr_SetString(PyExc_TypeError,"argument sizes must be equal");
      return NULL;
    }
  if ((PyArray_TYPE(a) == PyArray_COMPLEX64) && (PyArray_TYPE(b) == PyArray_FLOAT32))
    {
      a_data_sp = (npy_complex64*)PyArray_DATA(a);
      b_data_sp = (npy_float32*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp_sp = a_data_sp + i;
	  tmp_sp->real = b_data_sp[i] - tmp_sp->real * c; 
	}
    }
  else if ((PyArray_TYPE(a) == PyArray_COMPLEX128) && (PyArray_TYPE(b) == PyArray_FLOAT64))
    {
      a_data_dp = (npy_complex128*)PyArray_DATA(a);
      b_data_dp = (npy_float64*)PyArray_DATA(b);
      for (i=0; i<sz; ++i)
	{
	  tmp_dp = a_data_dp + i;
	  tmp_dp->real = b_data_dp[i] - tmp_dp->real * c; 
	}
    }
  else
    {
      PyErr_SetString(PyExc_TypeError,"argument types must be complex64 and float32");
      return NULL;
    }
  return Py_BuildValue("");
}

static
double m(double a, double b)
{
  if (a<0 && b<0)
    {
      if (a >= b) return a;
      return b;
    }
  if (a>0 && b>0)
    {
      if (a < b) return a;
      return b;
    }
  return 0.0;
}

static
__inline double hypot3(double a, double b, double c)
{
  return sqrt(a*a + b*b + c*c);
}

#define FLOAT32_EPS 0.0 //1e-8
#define FLOAT64_EPS 0.0 //1e-16

static PyObject *div_unit_grad(PyObject *self, PyObject *args)
{
  PyObject* f = NULL;
  npy_intp Nx, Ny, Nz;
  int i, j, k, im1, im2, ip1, jm1, jm2, jp1, km1, km2, kp1;
  npy_float64* f_data_dp = NULL;
  npy_float64* r_data_dp = NULL;
  npy_float32* f_data_sp = NULL;
  npy_float32* r_data_sp = NULL;
  double hx, hy, hz;
  double hx2, hy2, hz2;
  PyArrayObject* r = NULL;
  double fip, fim, fjp, fjm, fkp, fkm, fijk;
  double fimkm, fipkm, fjmkm, fjpkm, fimjm, fipjm, fimkp, fjmkp, fimjp;
  double aim, bjm, ckm, aijk, bijk, cijk;
  double Dxpf, Dxmf, Dypf, Dymf, Dzpf, Dzmf;
  double Dxma, Dymb, Dzmc;
  if (!PyArg_ParseTuple(args, "O(ddd)", &f, &hx, &hy, &hz))
    return NULL;
  hx2 = 2*hx;  hy2 = 2*hy;  hz2 = 2*hz;
  if (!PyArray_Check(f))
    {
      PyErr_SetString(PyExc_TypeError,"first argument must be array");
      return NULL;
    }
  if (PyArray_NDIM(f) != 3)
    {
      PyErr_SetString(PyExc_TypeError,"array argument must have rank 3");
      return NULL;
    }
  Nx = PyArray_DIM(f, 0);
  Ny = PyArray_DIM(f, 1);
  Nz = PyArray_DIM(f, 2);
  r = (PyArrayObject*)PyArray_SimpleNew(3, PyArray_DIMS(f), PyArray_TYPE(f));

  if (PyArray_TYPE(f) == PyArray_FLOAT32)
    {
      f_data_sp = (npy_float32*)PyArray_DATA(f);
      r_data_sp = (npy_float32*)PyArray_DATA(r);
      for (i=0; i<Nx; ++i)
	{
	  im1 = (i?i-1:0);
	  im2 = (im1?im1-1:0);
      	  ip1 = (i+1==Nx?i:i+1);
	  for (j=0; j<Ny; ++j)
	    {
	      jm1 = (j?j-1:0);
	      jm2 = (jm1?jm1-1:0);
	      jp1 = (j+1==Ny?j:j+1);
	      for (k=0; k<Nz; ++k)
		{
		  km1 = (k?k-1:0);
		  km2 = (km1?km1-1:0);
		  kp1 = (k+1==Nz?k:k+1);

		  fimjm = *((npy_float32*)PyArray_GETPTR3(f, im1, jm1, k));
		  fim = *((npy_float32*)PyArray_GETPTR3(f, im1, j, k));
		  fimkm = *((npy_float32*)PyArray_GETPTR3(f, im1, j, km1));
		  fimkp = *((npy_float32*)PyArray_GETPTR3(f, im1, j, kp1));
		  fimjp = *((npy_float32*)PyArray_GETPTR3(f, im1, jp1, k));

		  fjmkm = *((npy_float32*)PyArray_GETPTR3(f, i, jm1, km1));
		  fjm = *((npy_float32*)PyArray_GETPTR3(f, i, jm1, k));
		  fjmkp = *((npy_float32*)PyArray_GETPTR3(f, i, jm1, kp1));

		  fkm = *((npy_float32*)PyArray_GETPTR3(f, i, j, km1));
		  fijk = *((npy_float32*)PyArray_GETPTR3(f, i, j, k));
		  fkp = *((npy_float32*)PyArray_GETPTR3(f, i, j, kp1));

		  fjpkm = *((npy_float32*)PyArray_GETPTR3(f, i, jp1, km1));
		  fjp = *((npy_float32*)PyArray_GETPTR3(f, i, jp1, k));

		  fipjm = *((npy_float32*)PyArray_GETPTR3(f, ip1, jm1, k));
		  fipkm = *((npy_float32*)PyArray_GETPTR3(f, ip1, j, km1));
		  fip = *((npy_float32*)PyArray_GETPTR3(f, ip1, j, k));

		  Dxpf = (fip - fijk) / hx;
		  Dxmf = (fijk - fim) / hx;
		  Dypf = (fjp - fijk) / hy;
		  Dymf = (fijk - fjm) / hy;
		  Dzpf = (fkp - fijk) / hz;
		  Dzmf = (fijk - fkm) / hz;
		  aijk = hypot3(Dxpf, m(Dypf, Dymf), m(Dzpf, Dzmf));
		  bijk = hypot3(Dypf, m(Dxpf, Dxmf), m(Dzpf, Dzmf));
		  cijk = hypot3(Dzpf, m(Dypf, Dymf), m(Dxpf, Dxmf));

		  aijk = (aijk>FLOAT32_EPS?Dxpf / aijk:0.0);
		  bijk = (bijk>FLOAT32_EPS?Dypf / bijk: 0.0);
		  cijk = (cijk>FLOAT32_EPS?Dzpf / cijk:0.0); 
		  

		  Dxpf = (fijk - fim) / hx;
		  Dypf = (fimjp - fim) / hy;
		  Dymf = (fim - fimjm) / hy;
		  Dzpf = (fimkp - fim) / hz;
		  Dzmf = (fim - fimkm) / hz;
		  aim = hypot3(Dxpf, m(Dypf, Dymf), m(Dzpf, Dzmf));

		  aim = (aim>FLOAT32_EPS?Dxpf/aim:0.0); 


		  Dxpf = (fipjm - fjm) / hx;
		  Dxmf = (fjm - fimjm) / hx;
		  Dypf = (fijk - fjm) / hy;
		  Dzpf = (fjmkp - fjm) / hz;
		  Dzmf = (fjm - fjmkm) / hz;
		  bjm = hypot3(Dypf, m(Dxpf, Dxmf), m(Dzpf, Dzmf));

		  bjm = (bjm>FLOAT32_EPS?Dypf/bjm:0.0);
		  

		  Dxpf = (fipkm - fkm) / hx;
		  Dxmf = (fjm - fimkm) / hx;
		  Dypf = (fjpkm - fkm) / hy;
		  Dymf = (fkm - fjmkm) / hy;
		  Dzpf = (fijk - fkm) / hz;
		  ckm = hypot3(Dzpf, m(Dypf, Dymf), m(Dxpf, Dxmf));

		  ckm = (ckm>FLOAT32_EPS?Dzpf/ckm:0.0); 

		  Dxma = (aijk - aim) / hx;
		  Dymb = (bijk - bjm) / hy;
		  Dzmc = (cijk - ckm) / hz;
		  
		  //*((npy_float32*)PyArray_GETPTR3(r, i, j, k)) = Dxma/hx + Dymb/hy + Dzmc/hz;
		  *((npy_float32*)PyArray_GETPTR3(r, i, j, k)) = Dxma + Dymb + Dzmc;
		}
	    }
	}      
    }
  else if (PyArray_TYPE(f) == PyArray_FLOAT64)
    {
      f_data_dp = (npy_float64*)PyArray_DATA(f);
      r_data_dp = (npy_float64*)PyArray_DATA(r);
      for (i=0; i<Nx; ++i)
	{
	  im1 = (i?i-1:0);
	  im2 = (im1?im1-1:0);
      	  ip1 = (i+1==Nx?i:i+1);
	  for (j=0; j<Ny; ++j)
	    {
	      jm1 = (j?j-1:0);
	      jm2 = (jm1?jm1-1:0);
	      jp1 = (j+1==Ny?j:j+1);
	      for (k=0; k<Nz; ++k)
		{
		  km1 = (k?k-1:0);
		  km2 = (km1?km1-1:0);
		  kp1 = (k+1==Nz?k:k+1);

		  fimjm = *((npy_float64*)PyArray_GETPTR3(f, im1, jm1, k));
		  fim = *((npy_float64*)PyArray_GETPTR3(f, im1, j, k));
		  fimkm = *((npy_float64*)PyArray_GETPTR3(f, im1, j, km1));
		  fimkp = *((npy_float64*)PyArray_GETPTR3(f, im1, j, kp1));
		  fimjp = *((npy_float64*)PyArray_GETPTR3(f, im1, jp1, k));

		  fjmkm = *((npy_float64*)PyArray_GETPTR3(f, i, jm1, km1));
		  fjm = *((npy_float64*)PyArray_GETPTR3(f, i, jm1, k));
		  fjmkp = *((npy_float64*)PyArray_GETPTR3(f, i, jm1, kp1));

		  fkm = *((npy_float64*)PyArray_GETPTR3(f, i, j, km1));
		  fijk = *((npy_float64*)PyArray_GETPTR3(f, i, j, k));
		  fkp = *((npy_float64*)PyArray_GETPTR3(f, i, j, kp1));

		  fjpkm = *((npy_float64*)PyArray_GETPTR3(f, i, jp1, km1));
		  fjp = *((npy_float64*)PyArray_GETPTR3(f, i, jp1, k));

		  fipjm = *((npy_float64*)PyArray_GETPTR3(f, ip1, jm1, k));
		  fipkm = *((npy_float64*)PyArray_GETPTR3(f, ip1, j, km1));
		  fip = *((npy_float64*)PyArray_GETPTR3(f, ip1, j, k));

		  Dxpf = (fip - fijk) / hx;
		  Dxmf = (fijk - fim) / hx;
		  Dypf = (fjp - fijk) / hy;
		  Dymf = (fijk - fjm) / hy;
		  Dzpf = (fkp - fijk) / hz;
		  Dzmf = (fijk - fkm) / hz;
		  aijk = hypot3(Dxpf, m(Dypf, Dymf), m(Dzpf, Dzmf));
		  aijk = (aijk>FLOAT64_EPS?Dxpf / aijk:0.0);
		  bijk = hypot3(Dypf, m(Dxpf, Dxmf), m(Dzpf, Dzmf));
		  bijk = (bijk>FLOAT64_EPS?Dypf / bijk: 0.0);
		  cijk = hypot3(Dzpf, m(Dypf, Dymf), m(Dxpf, Dxmf));
		  cijk = (cijk>FLOAT64_EPS?Dzpf/cijk:0.0);

		  Dxpf = (fijk - fim) / hx;
		  Dypf = (fimjp - fim) / hy;
		  Dymf = (fim - fimjm) / hy;
		  Dzpf = (fimkp - fim) / hz;
		  Dzmf = (fim - fimkm) / hz;
		  aim = hypot3(Dxpf, m(Dypf, Dymf), m(Dzpf, Dzmf));
		  aim = (aim>FLOAT64_EPS?Dxpf/aim:0.0); 

		  Dxpf = (fipjm - fjm) / hx;
		  Dxmf = (fjm - fimjm) / hx;
		  Dypf = (fijk - fjm) / hy;
		  Dzpf = (fjmkp - fjm) / hz;
		  Dzmf = (fjm - fjmkm) / hz;
		  bjm = hypot3(Dypf, m(Dxpf, Dxmf), m(Dzpf, Dzmf));
		  bjm = (bjm>FLOAT64_EPS?Dypf/bjm:0.0);
		  


		  Dxpf = (fipkm - fkm) / hx;
		  Dxmf = (fjm - fimkm) / hx;
		  Dypf = (fjpkm - fkm) / hy;
		  Dymf = (fkm - fjmkm) / hy;
		  Dzpf = (fijk - fkm) / hz;
		  ckm = hypot3(Dzpf, m(Dypf, Dymf), m(Dxpf, Dxmf));
		  ckm = (ckm>FLOAT64_EPS?Dzpf/ckm:0.0); 
		  
		  Dxma = (aijk - aim) / hx;
		  Dymb = (bijk - bjm) / hy;
		  Dzmc = (cijk - ckm) / hz;

		  //*((npy_float64*)PyArray_GETPTR3(r, i, j, k)) = Dxma/hx + Dymb/hy + Dzmc/hz;
		  *((npy_float64*)PyArray_GETPTR3(r, i, j, k)) = Dxma + Dymb + Dzmc;
		}
	    }
	}
    }
  else
    {
      PyErr_SetString(PyExc_TypeError,"array argument type must be float64");
      return NULL;
    }
  return Py_BuildValue("N", r);
}

static PyObject *div_unit_grad1(PyObject *self, PyObject *args)
{
  PyObject* f = NULL;
  npy_intp Nx;
  int i, im1, im2, ip1;
  npy_float64* f_data_dp = NULL;
  npy_float64* r_data_dp = NULL;
  double hx;
  double hx2;
  PyArrayObject* r = NULL;
  double fip, fim, fijk;
  double aim, aijk;
  double Dxpf, Dxmf;
  double Dxma;
  if (!PyArg_ParseTuple(args, "Od", &f, &hx))
    return NULL;
  hx2 = 2*hx;
  if (!PyArray_Check(f))
    {
      PyErr_SetString(PyExc_TypeError,"first argument must be array");
      return NULL;
    }
  if (PyArray_NDIM(f) != 1)
    {
      PyErr_SetString(PyExc_TypeError,"array argument must have rank 1");
      return NULL;
    }
  Nx = PyArray_DIM(f, 0);
  r = (PyArrayObject*)PyArray_SimpleNew(1, PyArray_DIMS(f), PyArray_TYPE(f));

  if (PyArray_TYPE(f) == PyArray_FLOAT64)
    {
      f_data_dp = (npy_float64*)PyArray_DATA(f);
      r_data_dp = (npy_float64*)PyArray_DATA(r);
      for (i=0; i<Nx; ++i)
	{
	  im1 = (i?i-1:0);
      	  ip1 = (i+1==Nx?i:i+1);
	  fim = *((npy_float64*)PyArray_GETPTR1(f, im1));
	  fijk = *((npy_float64*)PyArray_GETPTR1(f, i));
	  fip = *((npy_float64*)PyArray_GETPTR1(f, ip1));
	  Dxpf = (fip - fijk) / hx;
	  //if (Dxpf==0.0) aijk = 0.0;
	  //else if (Dxpf<0.0) aijk = -1.0;
	  //else aijk = 1.0;
	  aijk = sqrt(Dxpf*Dxpf);
	  aijk = (aijk>FLOAT64_EPS?Dxpf / aijk:0.0);
	  //aijk = abs(Dxpf);
	  //aijk = (aijk>FLOAT64_EPS?Dxpf / aijk:0.0);
	  Dxpf = (fijk - fim) / hx;
	  //if (Dxpf==0.0) aim = 0.0;
	  //else if (Dxpf<0.0) aim = -1.0;
	  //else aim = 1.0;
	  //aim = abs(Dxpf);
	  //aim = (aim>FLOAT64_EPS?Dxpf/aim:0.0); 		  
	  aim = sqrt(Dxpf*Dxpf);
	  aim = (aim>FLOAT64_EPS?Dxpf/aim:0.0);
	  Dxma = (aijk - aim) / hx;
	  *((npy_float64*)PyArray_GETPTR1(r, i)) = Dxma;
	}
    }
  else
    {
      PyErr_SetString(PyExc_TypeError,"array argument type must be float64");
      return NULL;
    }
  return Py_BuildValue("N", r);
}

static PyObject *fourier_sphere(PyObject *self, PyObject *args)
{
  /*
    Computes Fourier Transform of an ellipsoid:

      fourier_sphere((Nx,Ny,Nz), (Dx, Dy, Dz), pcount) -> array

      Nx, Ny, Nz - defines the shape of an output array
      Dx, Dy, Dz - defines the diameters of the ellipsoid in index units
      pcount - defines periodicity parameter of the algorithm, the higher
        pcount is, the more accurate is the result but the more time it
	takes to compute the result. pcount should be around 1000.

    References:
      http://dx.doi.org/10.3247/SL2Math07.002
   */
  npy_intp dims[] = {0, 0, 0};
  double r[] = {1.0, 1.0, 1.0};
  double a, b;
  double ir, jr, kr;
  double sn, cs;
  int i,j,k,n1,n2,n3,nx,ny,nz;
  int n1_end, n2_end, n3_end;
  int dn1, dn2, dn3;
  int total, count, pcount;
  PyObject* result = NULL;
  double eps, inveps;
  clock_t start_clock = clock();
  double eta;
  PyObject* write_func = NULL;
  int verbose;
  if (!PyArg_ParseTuple(args, "(iii)(ddd)dO", &nx, &ny, &nz, r, r+1, r+2, &eps, &write_func))
    return NULL;
  if (write_func == Py_None)
    verbose = 0;
  else if (PyCallable_Check(write_func))
    verbose = 1;
  else
    {
      PyErr_SetString(PyExc_TypeError,"eighth argument must be None or callable object");
      return NULL;
    }
  dims[0] = nx;
  dims[1] = ny;
  dims[2] = nz;
  dn1 = nx;
  dn2 = ny;
  dn3 = nz;

#define min2(a,b) ((a>b)?(b):(a))
#define min3(a,b,c) min2((a), min2((b),(c)))
#define CALL_WRITE(ARGS) \
  if (verbose && PyObject_CallFunctionObjArgs ARGS == NULL) return NULL;

  r[0] /= dims[0];
  r[1] /= dims[1];
  r[2] /= dims[2];

  if (eps>1.0) // eps specifies peridicity count
    {
      eps = 2e-4 * pow(eps,-2.0/3.0) / min3(r[0]*r[0],r[1]*r[1],r[2]*r[2]);
      CALL_WRITE((write_func,PyUnicode_FromString("fourier_sphere: using eps = %.5e\n\n"), PyFloat_FromDouble(eps), NULL));
    }
  result = PyArray_SimpleNew(3, dims, PyArray_FLOAT64);
  r[0] *= M_PI;
  r[1] *= M_PI;
  r[2] *= M_PI;
  total = (dn1/2+1) * (dn2/2+1) * (dn3/2+1);
  count = 0;
  pcount = 0;
  inveps = 3.0/eps;
  nx = 1 + sqrt(inveps)/(dn1*r[0]);
  nx *= dn1;

  for (i=0; i<dn1/2+1; ++i)
    {
      for (j=0; j<dn2/2+1; ++j)
	{
	  for (k=0; k<dn3/2+1; ++k)
	    {
	      a = 0.0;
	      pcount = 0;
	      n1_end = i + nx;
	      for (n1=i-nx; n1<=n1_end; n1 += dn1)
		{
		  ir = n1*r[0];
		  ir *= ir;
		  if (ir>inveps) continue;
		  ny = 1 + sqrt(inveps-ir)/(dn2*r[1]);
		  ny *= dn2;
		  n2_end = j + ny;
		  for (n2=j-ny; n2<=n2_end; n2 += dn2)
		    {
		      jr = n2*r[1];
		      jr = jr*jr + ir;
		      if (jr>inveps) continue;
		      nz = (1 + sqrt(inveps-jr)/(dn3*r[2]));
		      nz *= dn3;
		      n3_end = k + nz;
		      for (n3=k-nz; n3<=n3_end; n3 += dn3)
			{
			  kr = n3*r[2];
			  kr = kr*kr + jr;
			  if (kr>inveps) continue;
			  if (kr==0.0)
			    a += 1.0;
			  else
			    {
			      b = sqrt(kr);
			      sincos(b, &sn, &cs);
			      a += 3.0 * (sn/b - cs)/kr;
			    }
			  pcount++;
			}
		    }
		}
	      *((npy_float64*)PyArray_GETPTR3(result, i, j, k)) = a;
	      if (i)
		*((npy_float64*)PyArray_GETPTR3(result, dn1-i, j, k)) = a;
	      if (j)
		*((npy_float64*)PyArray_GETPTR3(result, i, dn2-j, k)) = a;
	      if (k)
		*((npy_float64*)PyArray_GETPTR3(result, i, j, dn3-k)) = a;
	      if (i && j)
		*((npy_float64*)PyArray_GETPTR3(result, dn1-i, dn2-j, k)) = a;
	      if (i && k)
		*((npy_float64*)PyArray_GETPTR3(result, dn1-i, j, dn3-k)) = a;
	      if (j && k)
		*((npy_float64*)PyArray_GETPTR3(result, i, dn2-j, dn3-k)) = a;
	      if (i && j && k)
		*((npy_float64*)PyArray_GETPTR3(result, dn1-i, dn2-j, dn3-k)) = a;
	      count ++;
	    }
	  eta = (clock() - start_clock) * (total/(count+0.0)-1.0) / CLOCKS_PER_SEC;
	  CALL_WRITE((write_func,  
		      PyUnicode_FromString("\rfourier_sphere: %6.2f%% done (%d), ETA:%4.1fs"),
		      PyFloat_FromDouble((count*100.0)/total),
		      PyLong_FromLong(pcount),
		      PyFloat_FromDouble(eta),
		      NULL));
	}
    }
  *((npy_float64*)PyArray_GETPTR3(result, 0, 0, 0)) = 1.0; // normalize sum(sphere) to 1.0
  CALL_WRITE((write_func, PyUnicode_FromString("\n"), NULL));
  return Py_BuildValue("N", result);
}

char module_doc[] =
    "Python C extension module Richardson Lucy deconvolution algorithm.\n";


static PyMethodDef module_methods[] = {
  {"inverse_division_inplace",  inverse_division_inplace, METH_VARARGS, "inverse_division_inplace(a,b) == `a = b/a if a!=0 else 0`"},
  {"inverse_subtraction_inplace",  inverse_subtraction_inplace, METH_VARARGS, "inverse_subtraction_inplace(a,b,c) == `a = b-c*a`"},
  {"update_estimate_poisson", update_estimate_poisson, METH_VARARGS, "update_estimate_poisson(a,b,epsilon) -> e,s,u,n == `a *= b, s,u are photon counts`"},
  {"update_estimate_gauss", update_estimate_gauss, METH_VARARGS, "update_estimate_gauss(a,b,epsilon, alpha) -> e,s,u,n == `a += alpha * b, s,u are photon counts`"},
  {"div_unit_grad", div_unit_grad, METH_VARARGS, "div_unit_grad(f, (hx,hy,hz)) == `div(grad f/|grad f|)`"},
  {"div_unit_grad1", div_unit_grad1, METH_VARARGS, "div_unit_grad1(f, hx) == `div(grad f/|grad f|)`"},
  {"fourier_sphere", fourier_sphere, METH_VARARGS, "fourier_sphere((Nx, Ny, Nz), (Dx, Dy, Dz), eps or pcount)"},
  {"kullback_leibler_divergence", kullback_leibler_divergence, METH_VARARGS, "kullback_leibler_divergence(f, f0) -> float"},
  //  {"zero_if_zero_inplace", zero_if_zero_inplace, METH_VARARGS, "zero_if_zero_inplace(a,b) == `a = a if b!=0 else 0`"},
  //{"poisson_hist_factor_estimate", poisson_hist_factor_estimate, METH_VARARGS, "poisson_hist_factor_estimate(a,b,c) -> (stable,unstable)"},

  {NULL}  /* Sentinel */
};

//PyMODINIT_FUNC
//initops_ext(void)
//{
//  PyObject* m = NULL;
//  import_array();
//  if (PyErr_Occurred())
//    {PyErr_SetString(PyExc_ImportError, "can't initialize module ops_ext (failed to import numpy)"); return;}
//  m = Py_InitModule3("ops_ext", module_methods, "Provides operations in C.");
//}

#if PY_MAJOR_VERSION >= 3

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int module_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "ops_ext",
        NULL,
        sizeof(struct module_state),
        module_methods,
        NULL,
        module_traverse,
        module_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_ops_ext(void)

#else

#define INITERROR return

PyMODINIT_FUNC
initops_ext(void)
#endif
{
    PyObject *module;

    char *doc = (char *)PyMem_Malloc(sizeof(module_doc) + sizeof(_VERSION_));
    PyOS_snprintf(doc, sizeof(module_doc) + sizeof(_VERSION_),
                  module_doc, _VERSION_);

#if PY_MAJOR_VERSION >= 3
    moduledef.m_doc = doc;
    module = PyModule_Create(&moduledef);
#else
    module = Py_InitModule3("ops_ext", module_methods, doc);
#endif

    PyMem_Free(doc);

    if (module == NULL)
        INITERROR;

    if (_import_array() < 0) {
        Py_DECREF(module);
        INITERROR;
    }

    {
#if PY_MAJOR_VERSION < 3
    PyObject *s = PyUnicode_FromString(_VERSION_);
#else
    PyObject *s = PyUnicode_FromString(_VERSION_);
#endif
    PyObject *dict = PyModule_GetDict(module);
    PyDict_SetItemString(dict, "__version__", s);
    Py_DECREF(s);
    }

//    if (bessel_init() != 0) {
//        PyErr_Format(PyExc_ValueError, "bessel_init function failed");
//        Py_DECREF(module);
//        INITERROR;
//    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}