# -*- coding: utf-8 -*-
# psfgen.py

"""
# Copyright (c) 2007-2015, Christoph Gohlke
# Copyright (c) 2007-2015, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

Point spread function calculations for fluorescence microscopy.

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2015.03.19

Requirements
------------
* `CPython 2.7 or 3.4 <http://www.python.org>`_
* `Numpy 1.9 <http://www.numpy.org>`_
* `Psf.c 2015.03.19 <http://www.lfd.uci.edu/~gohlke/>`_
* `Matplotlib 1.4 <http://www.matplotlib.org>`_  (optional for plotting)

References
----------
(1) Electromagnetic diffraction in optical systems. II. Structure of the
    image field in an aplanatic system.
    B Richards and E Wolf. Proc R Soc Lond A, 253 (1274), 358-379, 1959.
(2) Focal volume optics and experimental artifacts in confocal fluorescence
    correlation spectroscopy.
    S T Hess, W W Webb. Biophys J (83) 2300-17, 2002.
(3) Electromagnetic description of image formation in confocal fluorescence
    microscopy.
    T D Viser, S H Wiersma. J Opt Soc Am A (11) 599-608, 1994.
(4) Photon counting histogram: one-photon excitation.
    B Huang, T D Perroud, R N Zare. Chem Phys Chem (5), 1523-31, 2004.
    Supporting information: Calculation of the observation volume profile.
(5) Gaussian approximations of fluorescence microscope point-spread function
    models.
    B Zhang, J Zerubia, J C Olivo-Marin. Appl. Optics (46) 1819-29, 2007.
(6) The SVI-wiki on 3D microscopy, deconvolution, visualization and analysis.
    http://support.svi.nl/wiki/NyquistRate
(7) Theory of Confocal Microscopy: Resolution and Contrast in Confocal
    Microscopy. http://www.olympusfluoview.com/theory/resolutionintro.html

Examples
--------
# >>> import _psf
# >>> args = dict(shape=(32,32), dims=(4,4), ex_wavelen=488, em_wavelen=520,
# ...             num_aperture=1.2, refr_index=1.333,
# ...             pinhole_radius=0.55, pinhole_shape='round')
# >>> obsvol = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL, **args)
# >>> print("%.5f, %.5f" % obsvol.sigma.ou)
# 2.58832, 1.37059
# >>> obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
# >>> obsvol[0, :3]
# array([ 1.     ,  0.51071,  0.04397])
# >>> # save the image plane to file
# >>> obsvol.slice(0).tofile("_test_slice.bin")
# >>> # save a full 3D PSF volume to file
# >>> obsvol.volume().tofile("_test_volume.bin")

"""



import math
import sys
import threading
import time

import numpy

try:
    from . import _psf
except ImportError:
    raise ImportError(
        "The compiled psf.c extension module could not be found. "
        "Psf.c can be obtained at http://www.lfd.uci.edu/~gohlke/")

from miplib.data.containers.image import Image

__version__ = '2015.03.19'
__docformat__ = 'restructuredtext en'
__all__ = 'PSF', 'Pinhole'

ANISOTROPIC = 1
ISOTROPIC = 2
GAUSSIAN = 4
GAUSSLORENTZ = 8
EXCITATION = 16
EMISSION = 32
WIDEFIELD = 64
CONFOCAL = 128
TWOPHOTON = 256
PARAXIAL = 512


class PsfFromFwhm(object):

    def __init__(self, fwhm, shape=(128, 128), dims=(4., 4.)):
        assert isinstance(fwhm, list)

        if len(fwhm) == 1:
            print ("Only one resolution value given. Assuming the same"
                   " resolution for the axial direction.")
            fwhm = [fwhm, ] * 2

        self.shape = int(shape[0]), int(shape[1])
        self.dims = Dimensions(px=shape, um=(float(dims[0]), float(dims[1])))

        self.spacing = list(x/y for x, y in zip(self.dims.um, self.dims.px))
        self.sigma_px = list(x/(2*math.sqrt(2*math.log(2))*y) for x, y in zip(fwhm, self.spacing))

        self.data = _psf.gaussian2d(self.dims.px, self.sigma_px)

    def xy(self):
        """Return a z slice of the PSF with rotational symmetries applied."""
        data = mirror_symmetry(_psf.zr2zxy(self.data))
        spacing = (self.spacing[1], self.spacing[1])

        center = self.shape[0] - 1
        return Image(data[center], spacing)

    def volume(self):
        """Return a 3D volume of the PSF with all symmetries applied.

        The shape of the returned array is
            (2*self.shape[0]-1, 2*self.shape[1]-1, 2*self.shape[1]-1)

        """
        data = mirror_symmetry(_psf.zr2zxy(self.data))
        spacing = (self.spacing[0], self.spacing[1], self.spacing[1])

        return Image(data, spacing)


class PSF(object):
    """Calculate point spread function of various types.

    Attributes
    ----------
    psftype : int
        A combination of the following properties. Valid combinations are
        listed in PSF.psftype.

        ANISOTROPIC or ISOTROPIC or GAUSSIAN or GAUSSLORENTZ
            Specify calculation model.
        EXCITATION or EMISSION or WIDEFIELD or CONFOCAL or TWOPHOTON
            Specify type of PSF.
        PARAXIAL
            Border case for Gaussian approximations.
    name : str
        A human readable label.
    data : 2D array of floats (C doubles)
        PSF values in z,r space normalized to the value at the origin.
    shape : sequence of int
        Size of the data array in pixel. Default (256, 256)
    dims : Dimension instance
        Dimensions of the data array in px (pixel), um (micrometers),
        ou (optical units), and au (airy units).
    ex_wavelen and em_wavelen : float or None
        Excitation or emission wavelengths in micrometers if applicable.
    num_aperture : float
        Numerical aperture (NA) of the objective. Default 1.2.
    refr_index : float
        Index of refraction of the sample medium. Default 1.333 (water).
    magnification : float
        Total magnification of the optical system. Default 1.0.
    underfilling : float
        Underfilling factor, i.e. the ratio of the radius of the objective
        back aperture to the exp(-2) radius of the excitation laser.
        Default 1.0.
    sigma : Dimension instance or None
        Gaussian sigma parameters in px (pixel), um (micrometers),
        ou (optical units), and au (airy units) if applicable.
    pinhole : Pinhole instance or None
        Pinhole applies to confocal types only.
    expsf, empsf : PSF instance or None
        Excitation or Emission PSF objects if applicable (i.e. when calculated
        intermediately for confocal)

    Notes
    -----
    Calculations of the isotropic PSFs are based on the complex integration
    representation for the diffraction near the image plane proposed by
    Richards and Wolf [1-4].

    Gaussian approximations are calculated according to [5].

    Widefield calculations are used if the pinhole radius is larger than ~8 au.

    Models for polarized excitation or emission light (ANISOTROPIC) and the
    Gaussian-Lorentzian approximation (GAUSSLORENTZ) are not implemented.

    """
    psftypes = {
        ISOTROPIC | EXCITATION: "Excitation, Isotropic",
        ISOTROPIC | EMISSION: "Emission, Isotropic",
        ISOTROPIC | WIDEFIELD: "Widefield, Isotropic",
        ISOTROPIC | CONFOCAL: "Confocal, Isotropic",
        ISOTROPIC | TWOPHOTON: "Two-Photon, Isotropic",
        GAUSSIAN | EXCITATION: "Excitation, Gaussian",
        GAUSSIAN | EMISSION: "Emission, Gaussian",
        GAUSSIAN | WIDEFIELD: "Widefield, Gaussian",  # == Gaussian Emission
        GAUSSIAN | CONFOCAL: "Confocal, Gaussian",
        GAUSSIAN | TWOPHOTON: "Two-Photon, Gaussian",
        GAUSSIAN | EXCITATION | PARAXIAL: "Excitation, Gaussian, Paraxial",
        GAUSSIAN | EMISSION | PARAXIAL: "Emission, Gaussian, Paraxial",
        GAUSSIAN | WIDEFIELD | PARAXIAL: "Widefield, Gaussian, Paraxial",
        GAUSSIAN | CONFOCAL | PARAXIAL: "Confocal, Gaussian, Paraxial",
        GAUSSIAN | TWOPHOTON | PARAXIAL: "Two-Photon, Gaussian, Paraxial",
    }

    def __init__(self, psftype, shape=(256, 256), dims=(4., 4.),
                 ex_wavelen=None, em_wavelen=None, num_aperture=1.2,
                 refr_index=1.333, magnification=1.0, underfilling=1.0,
                 pinhole_radius=None, pinhole_shape='round',
                 expsf=None, empsf=None, name=None):
        """Initialize the PSF object.

        Arguments
        ---------
        psftype, shape, num_aperture, refr_index, magnification, underfilling,
            expsf, and empsf:
            See PSF attributes.
        dims : sequence of float
            Dimensions of the data array in *micrometers*. Default (4., 4.)
        ex_wavelen and em_wavelen : float or None
            Excitation or emission wavelengths in *nanometers* if applicable.
        pinhole_radius : float or None
            Outer radius of the pinhole in *micrometers* in the object space.
            This is the back-projected radius, i.e. the real physical radius
            of the pinhole divided by the magnification of the system.
        pinhole_shape : str
            Either 'round' (default) or 'square'.

        """
        try:
            self.name = self.psftypes[psftype]
            self.psftype = psftype
        except Exception:
            raise ValueError("PSF type is invalid or not supported")

        if name:
            self.name = str(name)

        self.shape = int(shape[0]), int(shape[1])
        self.dims = Dimensions(px=shape, um=(float(dims[0]), float(dims[1])))

        self.ex_wavelen = ex_wavelen / 1e3 if ex_wavelen else None
        self.em_wavelen = em_wavelen / 1e3 if em_wavelen else None
        self.num_aperture = num_aperture
        self.refr_index = refr_index
        self.magnification = magnification
        self.underfilling = underfilling
        self.sigma = None
        self.pinhole = None
        self.expsf = expsf
        self.empsf = empsf

        if (not (psftype & EXCITATION)) and (em_wavelen is None):
            raise ValueError("emission wavelength not specified")

        if (not (psftype & EMISSION)) and (ex_wavelen is None):
            raise ValueError("excitation wavelength not specified")

        if (psftype & CONFOCAL) and (pinhole_radius is None):
            raise ValueError("pinhole radius not specified")

        self.sinalpha = self.num_aperture / self.refr_index
        if self.sinalpha >= 1.0:
            raise ValueError(
                "quotient of the numeric aperture (%.1f) and "
                "refractive index (%.1f) is greater than 1.0 (%.2f)" % (
                    self.num_aperture, self.refr_index, self.sinalpha))

        if psftype & EMISSION:
            au = (1.22 * self.em_wavelen / self.num_aperture)
            ou = zr2uv(self.dims.um, self.em_wavelen, self.sinalpha,
                       self.refr_index, self.magnification)
        else:
            au = (1.22 * self.ex_wavelen / self.num_aperture)
            ou = zr2uv(self.dims.um, self.ex_wavelen, self.sinalpha,
                       self.refr_index, 1.0)
        self.dims.update(ou=ou, au=(self.dims.um[0]/au, self.dims.um[1]/au))

        if pinhole_radius:
            self.pinhole = Pinhole(pinhole_radius, self.dims, pinhole_shape)

        start = time.clock()
        if psftype & GAUSSIAN:
            self.sigma = Dimensions(**self.dims)
            if self.underfilling != 1.0:
                raise NotImplementedError(
                    "underfilling not supported in Gaussian approximation")

            if psftype & EXCITATION or psftype & TWOPHOTON:
                widefield = True
                self.em_wavelen = None
                self.magnification = None
                self.pinh_radius = None
                lex = lem = self.ex_wavelen
                radius = 0.0
            elif psftype & EMISSION or psftype & WIDEFIELD:
                widefield = True
                self.ex_wavelen = None
                self.magnification = None
                lex = lem = self.em_wavelen
                radius = 0.0
            elif psftype & CONFOCAL:
                radius = self.pinhole.radius.um
                if radius > 9.76 * self.ex_wavelen/self.num_aperture:
                    # use widefield approximation for pinholes > 8 AU
                    widefield = True
                    lex = lem = self.ex_wavelen
                else:
                    widefield = False
                    lex = self.ex_wavelen
                    lem = self.em_wavelen
                if self.pinhole.shape != 'round':
                    raise NotImplementedError(
                        "Gaussian approximation only valid for round pinhole")

            paraxial = bool(psftype & PARAXIAL)
            self.sigma.um = _psf.gaussian_sigma(lex, lem, self.num_aperture,
                                                self.refr_index, radius,
                                                widefield, paraxial)
            self.data = _psf.gaussian2d(self.dims.px, self.sigma.px)

        elif psftype & ISOTROPIC:
            if psftype & EXCITATION or psftype & TWOPHOTON:
                self.em_wavelen = None
                self.magnification = None
                self.data = _psf.psf(0, self.shape, self.dims.ou, 1.0,
                                     self.sinalpha, self.underfilling, 1.0, 80)
            elif psftype & EMISSION:
                self.ex_wavelen = None
                self.underfilling = None
                self.data = _psf.psf(1, self.shape, self.dims.ou,
                                     self.magnification, self.sinalpha,
                                     1.0, 1.0, 80)
            elif psftype & CONFOCAL or psftype & WIDEFIELD:
                if em_wavelen < ex_wavelen:
                    raise ValueError("Excitation > Emission wavelength")
                # start threads to calculate excitation and emission PSF
                threads = []
                if not (self.expsf and
                        self.expsf.psftype == ISOTROPIC | EXCITATION):
                    threads.append((
                        "expsf",
                        PSFthread(ISOTROPIC | EXCITATION,
                                  shape, dims, ex_wavelen, None, num_aperture,
                                  refr_index, 1.0, underfilling)))
                if not (self.empsf and
                        self.empsf.psftype == ISOTROPIC | EMISSION):
                    threads.append((
                        "empsf",
                        PSFthread(ISOTROPIC | EMISSION,
                                  shape, dims, None, em_wavelen, num_aperture,
                                  refr_index, magnification, 1.0)))
                for a, t in threads:
                    t.start()
                for a, t in threads:
                    t.join()
                    setattr(self, a, t.psf)
                if not (self.expsf.iscompatible(self.empsf)):
                    raise ValueError(
                        "Excitation and Emission PSF not compatible")
                if psftype & WIDEFIELD or (self.pinhole.radius.um > 9.76 *
                                           self.ex_wavelen / self.num_aperture
                                           ):
                    # use widefield approximation for pinholes > 8 AU
                    self.data = _psf.obsvol(self.expsf.data, self.empsf.data)
                else:
                    self.data = _psf.obsvol(self.expsf.data, self.empsf.data,
                                            self.pinhole.kernel())

        if psftype & TWOPHOTON:
            self.data *= self.data
        self.time = float(time.clock()-start) * 1e3

    def __getitem__(self, key):
        """Return value of data array at position."""
        return self.data[key]

    def __str__(self):
        """Return properties of PSF object as string."""
        s = [self.name]
        s.append("  Shape: (%i, %i) pixel" % self.dims.px)
        s.append("  Dimensions: %s" % self.dims.format(
            ["um", "ou", "au"], ["%.2f", "%.2f", "%.2f"]))
        if self.ex_wavelen:
            s.append(
                "  Excitation Wavelength: %.1f nm" % (self.ex_wavelen * 1e3))
        if self.em_wavelen:
            s.append(
                "  Emission Wavelength: %.1f nm" % (self.em_wavelen * 1e3))
        s.append("  Numeric Aperture: %.2f" % self.num_aperture)
        s.append("  Refractive Index: %.2f" % self.refr_index)
        s.append("  Half Cone Angle: %.2f deg" % math.degrees(
            math.asin(self.sinalpha)))
        if self.magnification:
            s.append("  Magnification: %.2f" % self.magnification)
        if self.underfilling:
            s.append("  Underfilling: %.2f" % self.underfilling)
        if self.pinhole:
            s.append("  Pinhole Radius: %s" % self.pinhole.radius.format(
                ["um", "ou", "au", "px"], ["%.3f", "%.3f", "%.4f", "%.2f"]))
        if self.sigma:
            s.append("  Gauss Sigma: %s" % self.sigma.format(
                ["um", "ou", "au", "px"], ["%.3f", "%.3f", "%.3f", "%.2f"]))
        s.append("  Computing Time: %.2f ms\n" % self.time)
        return "\n".join(s)

    def iscompatible(self, other):
        """Return True if objects match dimensions and optical properties."""
        return ((self.dims.px[0] == other.dims.px[0])
                and (self.dims.px[1] == other.dims.px[1])
                and (self.dims.um[0] == other.dims.um[0])
                and (self.dims.um[1] == other.dims.um[1])
                and (self.num_aperture == other.num_aperture)
                and (self.refr_index == other.refr_index))

    def slice(self, key=slice(None)):
        """Return a z slice of the PSF with rotational symmetries applied."""
        return _psf.zr2zxy(self.data[key])

    def volume(self):
        """Return a 3D volume of the PSF with all symmetries applied.

        The shape of the returned array is
            (2*self.shape[0]-1, 2*self.shape[1]-1, 2*self.shape[1]-1)

        """
        return mirror_symmetry(_psf.zr2zxy(self.data))

    def imshow(self, subplot=111, **kwargs):
        """Log-plot PSF image using matplotlib.pyplot. Return plot axis."""
        title = kwargs.get("title", self.name)
        aspect = self.shape[1]/self.shape[0] * self.dims.um[0]/self.dims.um[1]
        kwargs.update(dict(data=self.data, title=title, subplot=subplot,
                           aspect=aspect))
        return imshow(**kwargs)

    def sted_correction(self, phi=3.22, sigma=6.0):
        """
        In STED PSF the regular confocal PSF gets modified by the depletion donut.
        This function does the corrections as eplained in:

        Zanella, R. et al., 2013. Towards real-time image deconvolution: application to
        confocal and STED microscopy. Scientific reports, 3, p.2523.
        Available at: http://www.nature.com/srep/2013/130828/srep02523/full/srep02523.htm.

        :param phi:   depletion gradient / um
        :param sigma: saturation factor Isted/Isat

        """
        phi *= self.dims.um[0]/self.dims.px[0]
        r_s = numpy.arange(0, self.data.shape[1], 1)
        multiplier = 1.0/(1+4*phi**2*sigma*r_s**2)

        self.data *= multiplier[None, :]



class PSFthread(threading.Thread):
    """Calculate point spread function in a thread."""

    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self)
        self.args = args
        self.kwargs = kwargs
        self.psf = None

    def run(self):
        self.psf = PSF(*self.args, **self.kwargs)


class Pinhole(object):
    """Pinhole object for confocal microscopy.

    Attributes
    ----------
    radius : Dimension instance
        Dimensions of the outer pinhole radius in px (pixel), um (micrometers),
        ou (optical units), and au (airy units).
    shape : str
        Shape of the pinhole. Either 'round' or 'square'.

    Examples
    --------
    >>> ph = Pinhole(0.1, dict(px=16, um=1.0), 'round')
    >>> print("%.5f" % ph.radius.px)
    1.60000
    >>> ph.kernel()
    array([[ 1.     ,  1.6    ,  0.6    ],
           [ 0.8    ,  1.18579,  0.36393],
           [ 0.3    ,  0.36393,  0.     ]])

    """
    shapes = {'round': 0, 'square': 4}

    def __init__(self, radius, dimensions, shape='round'):
        """Initialize the pinhole object.

        Arguments
        ---------
        radius : float
            Outer pinhole radius in micrometers in object space.
        dimensions : dict
            Dimensions of the object space in "px" (pixel), "um" (micrometers),
            "ou" (optical units), and "au" (airy units).
        shape : str
            Shape of the pinhole. Either 'round' (default) or 'square'.

        """
        self._corners = self.shapes[shape]
        self.shape = shape
        try:
            dimensions = dict((k, v[1]) for k, v in list(dimensions.items()))
        except TypeError:
            pass
        self.radius = Dimensions(**dimensions)
        self.radius.um = float(radius)
        self._kernel = None

    def __str__(self):
        s = [self.shape, "  Radius: %s" % str(self.radius)]
        return "\n".join(s)

    def kernel(self):
        """Return convolution kernel for integration over the pinhole."""
        if self._kernel is None:
            self._kernel = _psf.pinhole_kernel(self.radius.px, self._corners)
        return self._kernel


class Dimensions(dict):
    """Store dimensions in various units and perform linear conversions.

    Examples
    --------
    >>> dim = Dimensions(px=100, um=2)
    >>> dim(50, "px", "um")
    1.0
    >>> dim.px, dim.um
    (100, 2)
    >>> dim.px = 50
    >>> dim.um
    1.0
    >>> dim.format(("um", "px"), ("%.2f", "%.1f"))
    '1.00 um, 50.0 px'
    >>> dim = Dimensions(px=(100, 200), um=(2, 8))
    >>> dim((50, 50), "px", "um")
    (1.0, 2.0)
    >>> dim.ou = (1, 2)
    >>> dim.px
    (100, 200)
    >>> dim["px"] = (50, 100)
    >>> dim.ou
    (0.5, 1.0)

    """
    __slots__ = []

    def __call__(self, value, unit, newunit):
        """Return value given in unit in another unit."""
        dim = self[unit]
        new = self[newunit]
        try:
            return value * (new/dim)
        except TypeError:
            return tuple(v*(o/u) for v, u, o in zip(value, dim, new))

    def __setitem__(self, unit, value):
        """Add a dimension or rescale all dimensions to new value."""
        try:
            dim = self[unit]
        except KeyError:
            dict.__setitem__(self, unit, value)
        else:
            try:
                scale = value / dim
                for k, v in list(self.items()):
                    dict.__setitem__(self, k, v * scale)
            except TypeError:
                scale = tuple(v/d for v, d in zip(value, dim))
                for k, v in list(self.items()):
                    dict.__setitem__(
                        self, k, tuple(v*s for v, s in zip(self[k], scale)))

    def __getattr__(self, unit):
        """Return value of unit."""
        return self[unit]

    def __setattr__(self, unit, value):
        """Add a dimension or rescale all dimensions to new value."""
        self.__setitem__(unit, value)

    def format(self, keys, formatstr):
        """Return formatted string."""
        s = []
        try:
            for k, f in zip(keys, formatstr):
                s.append("%s %s" % (f % self[k], k))
        except TypeError:
            for k, f in zip(keys, formatstr):
                v = self[k]
                t = []
                for i in v:
                    t.append(f % i)
                s.append("(%s) %s" % (", ".join(t), k))
        return ", ".join(s)


def uv2zr(uv, wavelength, sinalpha, refr_index, magnification=1.0):
    """Return z,r in units of the wavelength from u,v given in optical units.

    For excitation, magnification should be 1.

    Examples
    --------
    >>> numpy.allclose(uv2zr((1, 1), 488, 0.9, 1.33),
    ...               (72.094692498695736, 64.885223248826165))
    True

    """
    a = wavelength / (2.0 * math.pi * sinalpha * refr_index * magnification)
    b = a / (sinalpha * magnification)
    return uv[0]*b, uv[1]*a


def zr2uv(zr, wavelength, sinalpha, refr_index, magnification=1.0):
    """Return u,v in optical units from z,r given in units of the wavelength.

    For excitation, magnification should be 1.

    Examples
    --------
    >>> numpy.allclose(zr2uv((1e3, 1e3), 488, 0.9, 1.33),
    ...               (13.870646580788051, 15.411829534208946))
    True

    """
    a = (2.0 * math.pi * refr_index * sinalpha * magnification) / wavelength
    b = a * sinalpha * magnification
    return zr[0]*b, zr[1]*a


def mirror_symmetry(data):
    """Apply mirror symmetry along one face in each dimension.

    The input array can be 1, 2 or 3-dimensional.

    The shape of the returned array is 2*data.shape-1 in each dimension.

    Examples
    --------
    >>> mirror_symmetry([0, 1])
    array([ 1.,  0.,  1.])
    >>> mirror_symmetry([[0, 1],[0, 1]])
    array([[ 1.,  0.,  1.],
           [ 1.,  0.,  1.],
           [ 1.,  0.,  1.]])
    >>> mirror_symmetry([[[0, 1],[0, 1]], [[0, 1],[0, 1]], [[0, 1],[0, 1]]])[0]
    array([[ 1.,  0.,  1.],
           [ 1.,  0.,  1.],
           [ 1.,  0.,  1.]])

    """
    data = numpy.array(data)
    result = numpy.empty([2*i-1 for i in data.shape], numpy.float64)
    if data.ndim == 1:
        x = data.shape[0] - 1
        result[x:] = data
        result[:x] = data[-1:0:-1]
    elif data.ndim == 2:
        x, y = (i-1 for i in data.shape)
        result[x:, y:] = data
        result[:x, y:] = data[-1:0:-1, :]
        result[:, :y] = result[:, -1:y:-1]
    elif data.ndim == 3:
        x, y, z = (i - 1 for i in data.shape)
        result[x:, y:, z:] = data
        result[:x, y:, z:] = data[-1:0:-1, :, :]
        result[:, :y, z:] = result[:, -1:y:-1, z:]
        result[:, :, :z] = result[:, :, -1:z:-1]
    else:
        raise NotImplementedError("%i-dimensional arrays not supported" %
                                  data.ndim)
    return result


def imshow(subplot, data, title=None, sharex=None, sharey=None,
           vmin=-2.5, vmax=0.0, cmap=None, interpolation='bilinear', **kwargs):
    """Log-plot image using matplotlib.pyplot. Return plot axis and plot.

    Mirror symmetry is applied along the x and y axes.

    Requires pyplot already imported ``from matplotlib import pyplot``.

    """
    pyplot = sys.modules['matplotlib.pyplot']

    ax = pyplot.subplot(subplot, sharex=sharex, sharey=sharey, axisbg='k')
    if title:
        pyplot.title(title)
    if cmap is None:
        cmap = pyplot.cm.cubehelix  # coolwarm
    # workaround: set alpha for i_bad
    cmap._init()
    cmap._lut[-1, -1] = 1.0

    im = pyplot.imshow(mirror_symmetry(numpy.log10(data)),
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       interpolation=interpolation, **kwargs)
    pyplot.axis('off')
    return ax, im


if __name__ == "__main__":
    import doctest
    numpy.set_printoptions(suppress=True, precision=5)
    doctest.testmod()