"""
A small utility for testing the performance of convolution operations.

The MKL optimizations on Anaconda seem to do miracles. I got several folds
speed-up in FFT, both in Numpy and Scipy. This will allow me to use easier
dependencies in my fusion code.

fftconvolve on Scipy seems to be the fastest of the alternatives that I tried.
"""
import accelerate.mkl as mkl
from numpy.fft import rfftn, irfftn
import numpy as np
from scipy.signal import fftconvolve

import time


def main():

    mkl.set_num_threads(mkl.get_max_threads())
    print "Am I using MKL FFT?: ", str(np.fft.using_mklfft)
    print "I am running on ", mkl.get_max_threads(), " threads"

    kuva = np.ones((1024, 1024, 320))
    kernel = np.ones((3, 3, 3))

    alku = time.time()
    fourier_convolve_fft(kuva, kernel)
    loppu = time.time()
    fourier_convolve_scipy(kuva, kernel)
    loppu2 = time.time()


    print "MKL FFT convolve took ", loppu-alku, " seconds"
    print "Scipy convolve took: ", loppu2-loppu, " seconds"

def fourier_convolve_fft(kuva, kernel):
    return irfftn(rfftn(kernel, s=kuva.shape)*rfftn(kuva))

def fourier_convolve_scipy(kuva, kernel):
    return fftconvolve(kuva, kernel, mode="same")

if __name__ == "__main__":
    main()

