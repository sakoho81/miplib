/* psf.c

A Python C extension module for calculating point spread functions.

Refer to the psf.py for documentation and tests.

:Authors:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_,
  Oliver Holub

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2015.03.19

Requirements
------------
* `CPython 2.7 or 3.4 <http://www.python.org>`_
* `Numpy 1.9 <http://www.numpy.org>`_
* A Python distutils compatible C compiler  (build)

Install
-------
Use this Python distutils setup script to build the extension module::

  # setup.py
  # Usage: ``python setup.py build_ext --inplace``
  from distutils.core import setup, Extension
  import numpy
  setup(name='_psf',
        ext_modules=[Extension('_psf', ['psf.c'],
                               include_dirs=[numpy.get_include()])])

License
-------
Copyright (c) 2007-2015, Christoph Gohlke
Copyright (c) 2007-2015, The Regents of the University of California
Produced at the Laboratory for Fluorescence Dynamics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holders nor the names of any
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#define _VERSION_ "2015.03.19"

#define WIN32_LEAN_AND_MEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "math.h"
#include "float.h"
#include "string.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#ifndef M_PI
#define M_PI (3.1415926535897932384626433832795)
#endif
#define M_2PI (6.283185307179586476925286766559)

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define SWAP(a, b) { register double t = (a); (a) = (b); (b) = t; }

#define BESSEL_LEN 1001 /* length of Bessel function lookup table */
#define BESSEL_RES 10.0 /* resolution of Bessel function lookup table */
#define BESSEL_INT 60   /* steps for integrating the Bessel integral */

/* lookup table for Bessel function of the first kind, orders 0, 1, and 2 */
double bessel_lut[BESSEL_LEN*3];

/*****************************************************************************/
/* C functions
*/

/*
Fast rounding of floating point to integer numbers.
*/

#ifdef WIN32
#ifndef _WIN64

int floor_int(double x)
{
    int i;
    const double _f = -0.5f;
    __asm {
        fld x
        fadd st, st(0)
        fadd _f
        fistp i
        sar i, 1
    }
    return i;
}

int ceil_int(double x)
{
    int i;
    const double _f = -0.5f;
    __asm {
        fld x
        fadd st, st(0)
        fsubr _f
        fistp i
        sar i, 1
    }
    return (-i);
}

#else
#define floor_int(x) (int)floor((double)(x))
#define ceil_int(x) (int)ceil((double)(x))
#endif
#else
#define floor_int(x) (int)floor((double)(x))
#define ceil_int(x) (int)ceil((double)(x))
#endif

/*
Return Bessel function at x for orders 0, 1, and 2.
Values are linearly interpolated from the lookup table.
*/
void bessel_lookup(double x, double* result)
{
    double* p;
    double alpha = x * BESSEL_RES;
    int index = floor_int(alpha);

    if (index < BESSEL_LEN) {
        p = &bessel_lut[index*3];
        alpha -= (double)index;
        result[0] = p[0] + alpha * (p[3] - p[0]);
        result[1] = p[1] + alpha * (p[4] - p[1]);
        result[2] = p[2] + alpha * (p[5] - p[2]);
    } else {
        result[0] = result[1] = result[2] = 0.0;
    }
}

/*
Initialize the global Bessel function lookup table.
Values are approximated by integrating Bessel's integral.
*/
int bessel_init(void)
{
    int i, j;
    double x, t, xst, dt, *ptr;

    memset(bessel_lut, 0, BESSEL_LEN*3*sizeof(double));

    dt = M_PI / (BESSEL_INT);
    ptr = bessel_lut;
    for (i = 0; i < BESSEL_LEN; i++) {
        x = -(double)i / BESSEL_RES;
        for (j = 0; j < BESSEL_INT; j++) {
            t = j * dt;
            xst = x * sin(t);
            ptr[0] += cos(xst);
            ptr[2] += cos(2.0*t + xst);
        }
        ptr[0] /= BESSEL_INT;
        ptr[2] /= BESSEL_INT;
        ptr += 3;
    }

    dt = M_PI / (BESSEL_INT-1);
    ptr = bessel_lut+1;
    for (i = 0; i < BESSEL_LEN; i++) {
        x = (double)i / BESSEL_RES;
        for (j = 0; j < BESSEL_INT; j++) {
            t = j * dt;
            *ptr += cos(t - x*sin(t));
        }
        *ptr /= BESSEL_INT-1;
        ptr += 3;
    }
    return 0;
}

/*
Calculate 2D Gaussian distribution.
*/
int gaussian2d(double *out, Py_ssize_t* shape, double* sigma)
{
    Py_ssize_t z, r;
    double sz, sr, t;

    if ((out == NULL) || (sigma[0] == 0) || (sigma[1] == 0))
        return -1;

    sz = -0.5 / (sigma[0]*sigma[0]);
    sr = -0.5 / (sigma[1]*sigma[1]);

    for (z = 0; z < shape[0]; z++) {
        t = z*z * sz;
        for (r = 0; r < shape[1]; r++) {
            *out++ = exp(t + r*r*sr);
        }
    }
    return 0;
}

/*
Calculate Gaussian parameters for the nonparaxial widefield case.
*/
void sigma_widefield(double* sz, double* sr, double nk, double cosa)
{
    double t = pow(cosa, 1.5);
    *sr = 1.0 / (nk * sqrt((4. - 7.*t + 3.*pow(cosa, 3.5)) / (7.*(1. - t))));
    *sz = (5.*sqrt(7.)*(1. - t)) / (nk * sqrt(6.*(4.*pow(cosa, 5) -
                     25.*pow(cosa, 3.5) + 42.*pow(cosa, 2.5) - 25.*t + 4.)));
}

/*
Calculate Gaussian parameters for point spread function approximation
according to B Zhang et al. Appl. Optics (46) 1819-29, 2007.
*/
int gaussian_sigma(double* sz, double* sr, double lex, double lem,
                   double NA, double n, double r, int widefield, int paraxial)
{
    if ((NA <= 0.0) || (n <= 0.0) || (lem <= 0.0) || ((NA/n) >= 1.0))
        return -1;

    if (widefield) {
        if (paraxial) {
            *sr = sqrt(2.) * lem / (M_2PI * NA);
            *sz = sqrt(6.) * lem / (M_2PI * NA*NA) * n * 2.;
        } else {
            sigma_widefield(sz, sr, n*M_2PI/lem, cos(asin(NA/n)));
        }
    } else {
        if ((r <= 0.0) || (lex <= 0.0))
            return -1;
        if (paraxial) {
            double kem = M_2PI / lem;
            double c1 = M_2PI / lex * r * NA;
            double c2 = M_2PI / lem * r * NA;
            double J0, J1, J[3];
            bessel_lookup(c2, J);
            J0 = J[0];
            J1 = J[1];
            *sr = sqrt(2./(c1*c1/(r*r) +
                (4.*c2*J0*J1 - 8.*J1*J1) / (r*r*(J0*J0 + J1*J1 - 1.))));
            *sz = 2.*sqrt(6./((c1*c1*NA*NA)/(r*r*n*n) -
                (48.*c2*c2*(J0*J0 + J1*J1) - 192.*J1*J1) /
                (n*n*kem*kem*r*r*r*r*(J0*J0 + J1*J1 - 1.))));
        } else {
            double e, sz_em, sr_em, sz_ex, sr_ex;
            double cosa = cos(asin(NA/n));
            sigma_widefield(&sz_ex, &sr_ex, n*M_2PI/lex, cosa);
            sigma_widefield(&sz_em, &sr_em, n*M_2PI/lem, cosa);
            e = sr_em*sr_em;
            e = 2.0 * e*e * (exp(r*r/(2.0*e)) - 1.0);
            *sr = sqrt((sr_ex*sr_ex * e) / (e + r*r * sr_ex*sr_ex));
            *sz = sz_ex*sz_em / sqrt(sz_ex*sz_ex + sz_em*sz_em);
        }
    }
    return 0;
}

/*
Apodization function for excitation.
*/
double apodization_excitation(double ct, double st, double _, double beta)
{
    return sqrt(ct) * exp(st*st * beta);
}

/*
Apodization function for isotropic fluorescence emission.
*/
double apodization_emission(double ct, double st, double M, double _)
{
    double t = M*st;
    return sqrt(ct / sqrt(1.0 - t*t));
}

/*
Calculate the Point Spread Function for unpolarized or circular polarized
light according to the diffraction proposed by Richards and Wolf.
See supporting information of B Huang et al. Chem Phys Chem (5) 1523-31, 2004.
*/
int psf(
    int type,        /* PSF type: 0: excitation, 1: emission */
    double *data,    /* output array[shape[0]][shape[1]] */
    Py_ssize_t* shape,      /* shape of data array */
    double* uvdim,   /* optical units in u and v dimension */
    double M,        /* lateral magnification factor */
    double sinalpha, /* numerical aperture / refractive index of medium */
    double beta,     /* underfilling ratio (1.0) */
    double gamma,    /* ex_wavelen / em_wavelen / refractive index (1.0) */
    int intsteps     /* number of steps for integrating over theta (50) */
    )
{
    Py_ssize_t i, j, k, u_shape, v_shape;
    double u, v, t, st, ct, re, im, re0, im0, re1, im1, re2, im2;
    double const0, const1, u_delta, v_delta, bessel[3];
    double alpha; /* integration over theta upper limit */
    double delta; /* step size for integrating over theta */
    double *cache, *cptr, *dptr;
    double (*apodization)(double, double, double, double);

    if ((intsteps < 4) || (sinalpha <= 0.0) || (sinalpha >= 1.0))
        return -1;

    switch (type) {
        case 0: /* excitation */
            apodization = apodization_excitation;
            alpha = asin(sinalpha);
            beta = -beta*beta / (sinalpha*sinalpha);
            gamma = M = 1.0;
            break;
        case 1: /* emission */
            apodization = apodization_emission;
            alpha = asin(sinalpha / M);
            beta = 1.0;
            break;
        default:
            return -1;
    }

    delta = alpha / (double)(intsteps-1);

    cache = cptr = (double *)malloc((intsteps*5)*sizeof(double));
    if (cache == NULL)
        return -1;

    const0 = gamma / sinalpha;
    const1 = gamma / (sinalpha * sinalpha);

    /* cache some values used in inner integration loop */
    for (k = 0; k < intsteps; k++) {
        t = k * delta;
        st = sin(t);
        ct = cos(t);
        t = st * apodization(ct, st, M, beta);
        cptr[0] = st * const0;
        cptr[1] = ct * const1;
        cptr[2] = t * (1.0 + ct);
        cptr[3] = t * st * (2.0); /* 4*I1(u,v) */
        cptr[4] = t * (1.0 - ct);
        cptr += 5;
    }

    u_shape = shape[0];
    v_shape = shape[1];
    u_delta = uvdim[0] / (double)(u_shape-1);
    v_delta = uvdim[1] / (double)(v_shape-1);
    dptr = data;

    for (i = 0; i < u_shape; i++) {
        u = u_delta * (double) i;
        for (j = 0; j < v_shape; j++) {
            v = v_delta * (double) j;
            re0 = im0 = re1 = im1 = re2 = im2 = 0.0;
            cptr = cache;
            /* integrate over theta using trapezoid rule */
            bessel_lookup(v * cptr[0], bessel);
            ct = u * cptr[1]; re = cos(ct); im = sin(ct);
            t = bessel[0]*cptr[2]*0.5; re0 += re*t; im0 += im*t;
            t = bessel[1]*cptr[3]*0.5; re1 += re*t; im1 += im*t;
            t = bessel[2]*cptr[4]*0.5; re2 += re*t; im2 += im*t;
            cptr += 5;
            for (k = 1; k < intsteps-1; k++) {
                bessel_lookup(v * cptr[0], bessel);
                ct = u * cptr[1];
                re = cos(ct); /* complex exponential with re=0 */
                im = sin(ct);
                t = bessel[0]*cptr[2]; re0 += re*t; im0 += im*t;
                t = bessel[1]*cptr[3]; re1 += re*t; im1 += im*t;
                t = bessel[2]*cptr[4]; re2 += re*t; im2 += im*t;
                cptr += 5;
            }
            bessel_lookup(v * cptr[0], bessel);
            ct = u * cptr[1]; re = cos(ct); im = sin(ct);
            t = bessel[0]*cptr[2]*0.5; re0 += re*t; im0 += im*t;
            t = bessel[1]*cptr[3]*0.5; re1 += re*t; im1 += im*t;
            t = bessel[2]*cptr[4]*0.5; re2 += re*t; im2 += im*t;

            *dptr++ = (re0*re0 + im0*im0 +
                       re1*re1 + im1*im1 +
                       re2*re2 + im2*im2);
        }
    }
    t = data[0];
    for (i = 0; i < u_shape*v_shape; i++)
        data[i] /= t;

    free(cache);
    return 0;
}

/*
Calculate the observation volume for unpolarized light by multiplying the
excitation PSF with the convolution of the emission PSF and detector kernel.

The PSFs and the detector kernel must have equal physical sizes per pixel.
The PSFs must be in zr space, the detector kernel in xy space.
*/
int obsvol(
    Py_ssize_t dimz,         /* size of PSF arrays in z dimension */
    Py_ssize_t dimr,         /* size of PSF arrays in r dimension */
    Py_ssize_t dimd,         /* pixel size of detector array (must be square) */
    double *obsvol,   /* output array[dimu, dimr, dimr] */
    double *ex_psf,   /* excitation PSF array[dimu, dimr] */
    double *em_psf,   /* emission PSF array[dimu, dimr] */
    double *detector) /* detector kernel array[dimd, dimd] or NULL */
{
    Py_ssize_t z, x, y, r, xx, i, ii, ri, index, indey, *_ri;
    double sum, rf, x2, t;
    double *exptr, *emptr, *ovptr, *_rf, *_em;
    Py_ssize_t _dimd = dimd;

    if (detector == NULL) {
        /* approximation for large pinholes == widefield */
        i = 0;
        for (z = 0; z < dimz; z++) {
            sum = em_psf[i] * M_PI * 0.25;
            ii = i;
            for (r = 1; r < dimr; r++) {
                sum += em_psf[++ii] * (double)r;
            }
            sum *= M_2PI;
            for (r = 0; r < dimr; r++, i++) {
                obsvol[i] = ex_psf[i] * sum;
            }
        }
    } else if (dimd < 2) {
        /* approximation for very small pinholes */
        for (i = 0; i < dimz*dimr; i++) {
            obsvol[i] = ex_psf[i] * em_psf[i];
        }
    } else {
        /* use detector/pinhole kernel */
        if (dimd > dimr) dimd = dimr;
        /* floor integer and remainder float of radius at xy */
        _ri = (Py_ssize_t *)malloc((dimr*dimd)*sizeof(Py_ssize_t));
        if (_ri == NULL)
            return -1;
        _rf = (double *)malloc((dimr*dimd)*sizeof(double));
        if (_rf == NULL) {
            free(_ri);
            return -1;
        }
        /* em_psf at xy */
        _em = (double *)malloc((dimr*dimd)*sizeof(double));
        if (_em == NULL) {
            free(_ri); free(_rf);
            return -1;
        }

        for (x = 0; x < dimd; x++) {
            x2 = (double)(x*x);
            indey = x;
            index = x*dimd;
            _ri[index] = _ri[indey] = x;
            _rf[index] = _rf[indey] = 0.0;
            for (y = 1; y <= x; y++) {
                index++;
                indey += dimd;
                rf = sqrt(x2 + y*y);
                ri = floor_int(rf);
                _ri[index] = _ri[indey] = (dimr > ri) ? ri : -1;
                _rf[index] = _rf[indey] = (dimr > ri+1) ? rf-(double)ri : 0.0;
            }
        }
        for (x = dimd; x < dimr; x++) {
            index = x*dimd;
            _ri[index] = x;
            _rf[index] = 0.0;
            x2 = (double)(x*x);
            for (y = 1; y < dimd; y++) {
                index++;
                rf = sqrt(x2 + y*y);
                ri = floor_int(rf);
                _ri[index] = (dimr > ri) ? ri : -1;
                _rf[index] = (dimr > ri+1) ? rf-(double)ri : 0.0;
            }
        }
        for (z = 0; z < dimz; z++) {
            exptr = &ex_psf[z*dimr];
            emptr = &em_psf[z*dimr];
            ovptr = &obsvol[z*dimr];
            /* emission psf in xy space */
            for (x = 0; x < dimd; x++) {
                indey = x;
                index = x*dimd;
                _em[index] = _em[indey] = emptr[x];
                for (y = 1; y <= x; y++) {
                    index++;
                    indey += dimd;
                    ri = _ri[index];
                    if (ri >= 0) {
                        rf = _rf[index];
                        _em[index] = _em[indey] =
                            rf ? emptr[ri]+rf*(emptr[ri+1]-emptr[ri])
                               : emptr[ri];
                    } else {
                        _em[index] = _em[indey] = 0.0;
                    }
                }
            }
            for (x = dimd; x < dimr; x++) {
                index = x*dimd;
                _em[index] = emptr[x];
                for (y = 1; y < dimd; y++) {
                    index++;
                    ri = _ri[index];
                    if (ri >= 0) {
                        rf = _rf[index];
                        _em[index] = rf ? emptr[ri]+rf*(emptr[ri+1]-emptr[ri])
                                        : emptr[ri];
                    } else {
                        _em[index] = 0.0;
                    }
                }
            }
            for (r = 0; r < dimr; r++) {
                /* Convolute emission PSF with detector kernel.
                For large kernels this is inefficient and should be
                replaced by a FFT based algorithm. */
                sum = 0.0;
                i = 1-dimd + (_dimd-dimd);
                for (x = r-dimd+1; x < MIN(r+dimd, dimr); x++) {
                    xx = abs(x) * dimd;
                    ii = abs(i++) * _dimd;
                    for (y = 0; y < dimd; y++) {
                        sum += _em[xx++] * detector[ii++];
                    }
                }
                /* multiply integral with excitation psf */
                ovptr[r] = sum * exptr[r];
            }
        }
        free(_ri);
        free(_rf);
        free(_em);
    }

    /* normalize maximum intensity */
    t = obsvol[0];
    for (i = 0; i < dimz*dimr; i++)
        obsvol[i] /= t;

    return 0;
}

/*
Calculate the detector kernel for integration over pinhole with trapezoid rule.

The radius denotes the outer radius, except for square shape, where it denotes
the inner radius.

*/
int pinhole_kernel(int corners, double* out, Py_ssize_t dim, double radius)
{
    Py_ssize_t i, j, k;
    double alpha, t;

    for (i = 0; i < dim*dim; i++)
        out[i] = 1.0;

    switch (corners) {
        case 0: /* round pinhole */
            /* fill corner */
            t = sqrt(2.0 * dim*dim) - dim;
            t = t / sqrt(2.0);
            k = dim - ceil_int(t);
            for (i = k; i < dim; i++)
                for (j = k; j < dim; j++)
                    out[i*dim+j] = 0.0;
            /* draw antialiased arc using eightfold symmetry */
            for (j = 0; j <= floor_int((dim-1)/sqrt(2)); j++) {
                k = ceil_int(sqrt(radius*radius - (double)(j*j)));
                alpha = 1.0 - (sqrt((double)(k*k + j*j)) - radius);
                alpha *= 0.5;
                out[k*dim+j] = out[j*dim+k] = alpha;
                if (k > 0) {
                    k--;
                    alpha = radius - sqrt((double)(k*k + j*j));
                    alpha = 0.5 + 0.5 * alpha;
                    out[k*dim+j] = out[j*dim+k] = alpha;
                }
                for (i = k+2; i < dim; i++) {
                    out[i*dim+j] = out[j*dim+i] = 0.0;
                }
            }
            break;
        case 4: /* square pinhole */
            alpha = 0.5 * (radius - (double)floor_int(radius));
            for (i = 0; i < dim; i++) {
                out[i*dim + dim-1] *= alpha;
                out[i + (dim-1)*dim] *= alpha;
            }
            alpha = 0.5 + alpha;
            for (i = 0; i < dim-1; i++) {
                out[i*dim + dim-2] *= alpha;
                out[i + (dim-2)*dim] *= alpha;
            }
            break;
        default:
            return -1;
    }
    for (i = 0; i < dim; i++)
        for (j = 1; j < dim; j++)
            out[i*dim+j] *= 2.0;

    return 0;
}

/*
Apply rotational symmetry around the 1st dimension.
*/
int zr2zxy(double* data, double* out, Py_ssize_t dimz, Py_ssize_t dimr)
{
    Py_ssize_t x, y, z, x2, ri, index, indey, *_ri;
    double rf, *_rf, *dptr, *optr;

    /* floor integer and remainder float fraction of radius at xy */
    _ri = (Py_ssize_t *)malloc((dimr*dimr)*sizeof(Py_ssize_t));
    if (_ri == NULL)
        return -1;
    _rf = (double *)malloc((dimr*dimr)*sizeof(double));
    if (_rf == NULL) {
        free(_ri);
        return -1;
    }

    for (x = 0; x < dimr; x++) {
        x2 = x*x;
        indey = x;
        index = x*dimr;
        for (y = 0; y <= x; y++) {
            rf = sqrt((double)(x2 + y*y));
            ri = floor_int(rf);
            _ri[index] = _ri[indey] = (dimr > ri) ? ri : -1;
            _rf[index] = _rf[indey] = (dimr > ri+1) ? rf-(double)ri : 0.0;
            index++;
            indey += dimr;
        }
    }
    for (z = 0; z < dimz; z++) {
        dptr = &data[z*dimr];
        optr = &out[z*dimr*dimr];
        for (x = 0; x < dimr; x++) {
            indey = x;
            index = x*dimr;
            optr[index] = optr[indey] = dptr[x];
            for (y = 1; y <= x; y++) {
                index++;
                indey += dimr;
                ri = _ri[index];
                if (ri >= 0) {
                    rf = _rf[index];
                    optr[index] = optr[indey] =
                        rf ? dptr[ri]+rf*(dptr[ri+1]-dptr[ri]) : dptr[ri];
                } else {
                    optr[index] = optr[indey] = 0.0;
                }
            }
        }
    }
    free(_ri);
    free(_rf);
    return 0;
}

/*****************************************************************************/
/* Python functions */

/*
Numpy array converter for use with PyArg_Parse functions.
Ensures that array is C contiguous and data type is double.
*/
static int PyDoubleArray_Converter(PyObject *object, PyObject **address)
{
    if (PyArray_Check(object) &&
        (PyArray_TYPE((PyArrayObject *)object) == NPY_DOUBLE) &&
        (PyArray_FLAGS((PyArrayObject *)object) & NPY_ARRAY_C_CONTIGUOUS)) {
        *address = object;
        Py_INCREF(object);
        return NPY_SUCCEED;
    } else {
        *address = PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_ALIGNED);
        if (*address == NULL)
            return NPY_FAIL;
        return NPY_SUCCEED;
    }
}

/*
Return ith element of a Python sequence as long, or 0 on failure.
*/
long PySequence_GetInteger(PyObject *obj, Py_ssize_t i)
{
    long value;
    PyObject *item = PySequence_GetItem(obj, i);
    if (item == NULL ||
#if PY_MAJOR_VERSION < 3
        !PyInt_Check(item)
#else
        !PyLong_Check(item)
#endif
        ) {
        PyErr_Format(PyExc_ValueError, "expected integer number");
        Py_XDECREF(item);
        return 0;
    }

#if PY_MAJOR_VERSION < 3
    value = PyInt_AsLong(item);
#else
    value = PyLong_AsLong(item);
#endif
    Py_XDECREF(item);
    return value;
}

/*
Return the ith element of a Python sequence as double, or 0.0 on failure.
*/
double PySequence_GetDouble(PyObject *obj, Py_ssize_t i)
{
    PyObject *item = NULL;
    double value;
    item = PySequence_GetItem(obj, i);
    if (item == NULL || !PyNumber_Check(item)) {
        Py_XDECREF(item);
        PyErr_Format(PyExc_ValueError, "expected floating point number");
        return 0.0;
    }
    value = PyFloat_AsDouble(item);
    Py_XDECREF(item);
    return value;
}

/*
Python wrapper function for the psf() function.
*/
char py_psf_doc[] =
    "Return the point spread function for unpolarized light in z,r space.";

static PyObject* py_psf(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    PyObject *pyshape = NULL;
    PyObject *pyuvdim = NULL;
    PyArrayObject *out = NULL;
    Py_ssize_t shape[2];
    int type, error;
    double sinalpha, mag, uvdim[2];
    double beta = 1.0;
    double gamma = 1.0;
    int intsteps = 50;

    static char *kwlist[] = {"type", "shape", "uvdim", "magnification",
                             "sinalpha", "beta", "gamma", "intsteps", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iOOdd|ddi", kwlist,
        &type, &pyshape, &pyuvdim, &mag, &sinalpha, &beta, &gamma, &intsteps))
        goto _fail;

    Py_INCREF(pyshape);
    Py_INCREF(pyuvdim);

    if ((type < 0) || (type > 1)) {
        PyErr_Format(PyExc_ValueError, "type is not 0 or 1");
        goto _fail;
    }

    if (!(PySequence_Check(pyshape) && PySequence_Size(pyshape) == 2)) {
        PyErr_Format(PyExc_ValueError, "shape is not a sequence of length 2");
        goto _fail;
    }

    if (!(PySequence_Check(pyuvdim) && PySequence_Size(pyuvdim) == 2)) {
        PyErr_Format(PyExc_ValueError, "uvdim is not a sequence of length 2");
        goto _fail;
    }

    if ((sinalpha <= 0.0) || (sinalpha >= 1.0)) {
        PyErr_Format(PyExc_ValueError, "sinalpha is not in interval ]0, 1[");
        goto _fail;
    }

    if (mag <= 0.0) {
        PyErr_Format(PyExc_ValueError, "magnification is smaller than 0");
        goto _fail;
    }

    if (intsteps < 3) {
        PyErr_Format(PyExc_ValueError, "less than 3 integration steps");
        goto _fail;
    }

    shape[0] = PySequence_GetInteger(pyshape, 0);
    shape[1] = PySequence_GetInteger(pyshape, 1);
    uvdim[0] = PySequence_GetDouble(pyuvdim, 0);
    uvdim[1] = PySequence_GetDouble(pyuvdim, 1);

    out = (PyArrayObject*)PyArray_ZEROS(2, shape, NPY_DOUBLE, 0);
    if (out == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate array");
        goto _fail;
    }

    Py_BEGIN_ALLOW_THREADS
    error = psf(type, (double *)PyArray_DATA(out), shape, uvdim,
                mag, sinalpha, beta, gamma, intsteps);
    Py_END_ALLOW_THREADS

    if (error != 0) {
        PyErr_Format(PyExc_ValueError, "psf() function failed");
        goto _fail;
    }

    Py_DECREF(pyshape);
    Py_DECREF(pyuvdim);
    return PyArray_Return(out);

  _fail:
    Py_XDECREF(pyshape);
    Py_XDECREF(pyuvdim);
    Py_XDECREF(out);
    return NULL;
}

/*
Python wrapper function for the obsvol() function.
*/
char py_obsvol_doc[] =
    "Return the observation volume for one photon excitation in z,r space.";

static PyObject* py_obsvol(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    PyArrayObject *ex_psf = NULL;
    PyArrayObject *em_psf = NULL;
    PyArrayObject *detector = NULL;
    PyArrayObject *out = NULL;
    int error;

    static char *kwlist[] = {"ex_psf", "em_psf", "detector", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|O&", kwlist,
        PyDoubleArray_Converter, &ex_psf,
        PyDoubleArray_Converter, &em_psf,
        PyDoubleArray_Converter, &detector))
        goto _fail;

    if (PyArray_NDIM(ex_psf) == 3 || PyArray_NDIM(em_psf) == 3) {
        PyErr_Format(PyExc_NotImplementedError,
            "three dimensional PSF are not supported");
        goto _fail;
    }

    if (PyArray_NDIM(ex_psf) != 2 || PyArray_NDIM(em_psf) != 2) {
        PyErr_Format(PyExc_ValueError,
            "not all PSF arrays are 2 dimensional");
        goto _fail;
    }

    if (PyArray_DIM(ex_psf, 0) != PyArray_DIM(em_psf, 0) ||
        PyArray_DIM(ex_psf, 1) != PyArray_DIM(em_psf, 1)) {
        PyErr_Format(PyExc_ValueError, "PSF arrays are not same size");
        goto _fail;
    }

    if ((detector) && (PyArray_NDIM(detector)!= 2 ||
        PyArray_DIM(detector, 0) != PyArray_DIM(detector, 1))) {
        PyErr_Format(PyExc_ValueError, "detector kernel is not square");
        goto _fail;
    }

    out = (PyArrayObject*)PyArray_ZEROS(2, PyArray_DIMS(ex_psf),
                                        NPY_DOUBLE, 0);
    if (out == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate array");
        goto _fail;
    }

    error = obsvol(
        (int)PyArray_DIM(ex_psf, 0),
        (int)PyArray_DIM(ex_psf, 1),
        detector ? (int)PyArray_DIM(detector, 0) : 0,
        (double *)PyArray_DATA(out),
        (double *)PyArray_DATA(ex_psf),
        (double *)PyArray_DATA(em_psf),
        detector ? (double *)PyArray_DATA(detector) : NULL);

    if (error != 0) {
        PyErr_Format(PyExc_ValueError, "obsvol() function failed");
        goto _fail;
    }

    Py_DECREF(ex_psf);
    Py_DECREF(em_psf);
    Py_XDECREF(detector);
    return PyArray_Return(out);

  _fail:
    Py_XDECREF(ex_psf);
    Py_XDECREF(em_psf);
    Py_XDECREF(detector);
    Py_XDECREF(out);
    return NULL;
}

/*
Python wrapper function for accessing the Bessel lookup table.
*/
char py_bessel_doc[] =
    "Return the lookup table for the Bessel function of orders 0, 1, 2.";

static PyObject* py_bessel(PyObject *obj)
{
    PyArrayObject *out = NULL;
    Py_ssize_t shape[] = {BESSEL_LEN, 3};
    out = (PyArrayObject*)PyArray_SimpleNewFromData(2, shape,
                                                    NPY_DOUBLE, bessel_lut);
    if (out == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate array");
        goto _fail;
    }

    return PyArray_Return(out);

  _fail:
    Py_XDECREF(out);
    return NULL;
}

/*
Python wrapper function for the zr2zxy() function.
*/
char py_zr2zxy_doc[] =
    "Return new array with rotational symmetry applied in 1st dimension.";

static PyObject* py_zr2zxy(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    PyArrayObject *data = NULL;
    PyArrayObject *out = NULL;
    int error, ndims;
    Py_ssize_t shape[3];

    static char *kwlist[] = {"data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist,
        PyDoubleArray_Converter, &data))
        goto _fail;

    if ((PyArray_NDIM(data) != 1) && (PyArray_NDIM(data) != 2)) {
        PyErr_Format(PyExc_ValueError,
            "input array is not 1 or 2 dimensional");
        goto _fail;
    }

    if (PyArray_NDIM(data) == 1) {
        ndims = 2;
        shape[0] = shape[1] = PyArray_DIM(data, 0);
    } else {
        ndims = 3;
        shape[0] = PyArray_DIM(data, 0);
        shape[1] = shape[2] = PyArray_DIM(data, 1);
    }
    out = (PyArrayObject*)PyArray_ZEROS(ndims, shape, NPY_DOUBLE, 0);
    if (out == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate array");
        goto _fail;
    }

    error = zr2zxy((double *)PyArray_DATA(data),
                   (double *)PyArray_DATA(out),
                   (ndims == 3) ? (int)shape[0] : 1, (int)shape[1]);

    if (error != 0) {
        PyErr_Format(PyExc_ValueError, "zr2zxy() function failed");
        goto _fail;
    }

    Py_DECREF(data);
    return PyArray_Return(out);

  _fail:
    Py_XDECREF(data);
    Py_XDECREF(out);
    return NULL;
}

/*
Python wrapper for the pinhole_kernel() function.
*/
char py_pinhole_kernel_doc[] =
    "Return kernel for integrating over pinhole.";

static PyObject* py_pinhole_kernel(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    PyArrayObject *out = NULL;
    int error;
    int corners = 0;
    int dim = 0;
    double radius;
    Py_ssize_t shape[2];

    static char *kwlist[] = {"radius", "corners", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|i", kwlist,
        &radius, &corners))
        goto _fail;

    if ((corners != 0) && (corners != 4)) {
        PyErr_Format(PyExc_ValueError,
            "pinhole shape not supported: %i", corners);
        goto _fail;
    }

    /* for squares, reduce radius to inner radius */
    if (corners == 4)
        radius /= sqrt(2.0);

    dim = ceil_int(radius) + 1;
    shape[0] = shape[1] = dim;
    out = (PyArrayObject*)PyArray_ZEROS(2, shape, NPY_DOUBLE, 0);
    if (out == NULL)  {
        PyErr_Format(PyExc_MemoryError, "unable to allocate array");
        goto _fail;
    }

    Py_BEGIN_ALLOW_THREADS
    error = pinhole_kernel(corners, (double *)PyArray_DATA(out), dim, radius);
    Py_END_ALLOW_THREADS

    if (error != 0) {
        PyErr_Format(PyExc_ValueError, "pinhole_kernel() function failed");
        goto _fail;
    }

    return PyArray_Return(out);

  _fail:
    Py_XDECREF(out);
    return NULL;
}

/*
Python wrapper for the gaussian2d() function.
*/
char py_gaussian2d_doc[] =
    "Return the 2D Gaussian in z,r space.";

static PyObject* py_gaussian2d(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    PyArrayObject *out = NULL;
    PyObject *pyshape = NULL;
    PyObject *pysigma = NULL;
    Py_ssize_t shape[2];
    double sigma[2];
    int error;
    static char *kwlist[] = {"shape", "sigma", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist,
        &pyshape, &pysigma))
        goto _fail;

    Py_INCREF(pyshape);
    Py_INCREF(pysigma);

    if (!(PySequence_Check(pyshape) && PySequence_Size(pyshape) == 2 &&
          PySequence_Check(pysigma) && PySequence_Size(pysigma) == 2)) {
        PyErr_Format(PyExc_ValueError,
            "input parameters must be sequences of length 2");
        goto _fail;
    }

    shape[0] = PySequence_GetInteger(pyshape, 0);
    shape[1] = PySequence_GetInteger(pyshape, 1);
    out = (PyArrayObject*)PyArray_ZEROS(2, shape, NPY_DOUBLE, 0);
    if (out == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate array");
        goto _fail;
    }

    sigma[0] = PySequence_GetDouble(pysigma, 0);
    sigma[1] = PySequence_GetDouble(pysigma, 1);

    Py_BEGIN_ALLOW_THREADS
    error = gaussian2d((double *)PyArray_DATA(out), &shape[0], &sigma[0]);
    Py_END_ALLOW_THREADS

    if (error != 0) {
        PyErr_Format(PyExc_ValueError, "gaussian2d() function failed");
        goto _fail;
    }

    Py_DECREF(pyshape);
    Py_DECREF(pysigma);
    return PyArray_Return(out);

  _fail:
    Py_XDECREF(pyshape);
    Py_XDECREF(pysigma);
    Py_XDECREF(out);
    return NULL;
}

/*
Python wrapper for the gaussian_sigma() function.
*/
char py_gaussian_sigma_doc[] =
    "Return Gaussian sigma parameters for PSF approximation.";

static PyObject* py_gaussian_sigma(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    PyObject *widefieldobj = NULL;
    PyObject *paraxialobj = NULL;
    int error;
    int paraxial = 0;
    int widefield = 1;
    double radius = 1.0;
    double lex, lem, num_aperture, refr_index, sigmaz, sigmar;

    static char *kwlist[] = {"lex", "lem", "num_aperture", "refr_index",
                             "pinhole_radius", "widefield", "paraxial", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dddd|dOO", kwlist,
        &lex, &lem, &num_aperture, &refr_index, &radius,
        &widefieldobj, &paraxialobj))
        return NULL;

    if (widefieldobj != NULL)
        widefield = PyObject_IsTrue(widefieldobj);

    if (paraxialobj != NULL)
        paraxial = PyObject_IsTrue(paraxialobj);

    error = gaussian_sigma(&sigmaz, &sigmar, lex, lem, num_aperture,
                           refr_index, radius, widefield, paraxial);

    if (error != 0) {
        PyErr_Format(PyExc_ValueError, "gaussian_sigma() function failed");
        goto _fail;
    }

    return Py_BuildValue("(d,d)", sigmaz, sigmar);

  _fail:
    return NULL;
}

/*****************************************************************************/
/* Python module */

char module_doc[] =
    "Python C extension module for calculating point spread functions.\n\n"
    "Refer to the associated psf.py module for documentation.\n";

static PyMethodDef module_methods[] = {
    {"psf", (PyCFunction)py_psf,
        METH_VARARGS|METH_KEYWORDS, py_psf_doc},
    {"obsvol", (PyCFunction)py_obsvol,
        METH_VARARGS|METH_KEYWORDS, py_obsvol_doc},
    {"pinhole_kernel", (PyCFunction)py_pinhole_kernel,
        METH_VARARGS|METH_KEYWORDS, py_pinhole_kernel_doc},
    {"zr2zxy", (PyCFunction)py_zr2zxy,
        METH_VARARGS|METH_KEYWORDS, py_zr2zxy_doc},
    {"gaussian2d", (PyCFunction)py_gaussian2d,
        METH_VARARGS|METH_KEYWORDS, py_gaussian2d_doc},
    {"gaussian_sigma", (PyCFunction)py_gaussian_sigma,
        METH_VARARGS|METH_KEYWORDS, py_gaussian_sigma_doc},
    {"bessel_lut", (PyCFunction)py_bessel,
        METH_NOARGS, py_bessel_doc},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

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
        "_psf",
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
PyInit__psf(void)

#else

#define INITERROR return

PyMODINIT_FUNC
init_psf(void)
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
    module = Py_InitModule3("_psf", module_methods, doc);
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
    PyObject *s = PyString_FromString(_VERSION_);
#else
    PyObject *s = PyUnicode_FromString(_VERSION_);
#endif
    PyObject *dict = PyModule_GetDict(module);
    PyDict_SetItemString(dict, "__version__", s);
    Py_DECREF(s);
    }

    if (bessel_init() != 0) {
        PyErr_Format(PyExc_ValueError, "bessel_init function failed");
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}