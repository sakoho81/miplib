import os
import sys
import time

from pyculib import cuda_compatible
import miplib.processing.to_string as ops_output
import miplib.ui.utils as uiutils
from miplib.processing.deconvolution import deconvolve_cuda, deconvolve
from miplib.psf import psfgen
from miplib.ui.cli import miplib_entry_point_options as options
from miplib.data.io import read as imread


def main():
    args = options.get_deconvolve_script_options(sys.argv[1:])

    # Load image
    image_name = args.image
    dir = args.working_directory
    image_path = os.path.join(dir, image_name)

    # Figure out an image type and use the correct loader.
    if image_path.endswith('.tif'):
        image = imread.get_image(image_path)
    elif image_path.endswith('.mat'):
        image = imread.get_image(image_path)[args.carma_gate_idx, args.carma_det_idx]
    else:
        raise AttributeError("Unknown image type %s" % image_path.split('.')[-1])


    # Load/Generate PSF
    if args.psf == "estimate":
        psf = psfgen.PSF(psftype=args.psf_type, shape=args.psf_shape, dims=args.psf_size,
                         ex_wavelen=args.ex_wl, em_wavelen=args.em_wl, num_aperture=args.na,
                         refr_index=args.refractive_index, magnification=args.magnification,
                         pinhole_radius=args.pinhole_radius)

        if args.sted_psf:
            psf.sted_correction(args.sted_phi, args.sted_sigma)

            psf_image = psf.volume()
            psf_spacing = psf.dims['um'][0]/psf.shape[0]

            if image.get_dimensions()[0] == 1:
                psf_image = psf_image[psf_image.shape/2]

            psf = image.Image(psf_image, psf_spacing)

    else:
        psf = image.Image.get_image_from_imagej_tiff(args.psf_path)

    # Start deconvolution
    if cuda_compatible():
        print("Found a compatible GPU. The image deconvolution will be run with " \
              "hardware acceleration.")
        task = deconvolve_cuda.DeconvolutionRLCuda(image, psf, args)
    else:
        task = deconvolve.DeconvolutionRL(image, psf, args)

    begin = time.time()
    task.execute()
    end = time.time()
    result = task.get_result()

    print("Deconvolution complete.")
    print("The deconvolution process with %i iterations " \
          "took %s (H:M:S) to complete." % (args.max_nof_iterations,
                                            ops_output.format_time_string(
                                                end - begin)))
    if uiutils.get_user_input("Do you want to save the result to TIFF? "):
        file_path = os.path.join(args.working_directory,
                                 "fusion_result.tif")
        result.save_to_tiff(file_path)


if __name__ == "__main__":
        main()

