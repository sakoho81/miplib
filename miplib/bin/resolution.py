import datetime
import os
import sys


import numpy as np
import pandas
import pydeconvolution.frc.fourier_ring_correlation as frcmeas
import pydeconvolution.io.myimage as myimage
import pydeconvolution.options as options
import pydeconvolution.utils.str_utils as strutils


def main():

    # Get input arguments
    args = options.get_frc_script_options(sys.argv[1:])
    path = args.working_directory

    # Create output directory
    output_dir = datetime.datetime.now().strftime("%Y-%m-%d") + '_PyIQ_FRC_output'
    output_dir = os.path.join(args.working_directory, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    date_now = datetime.datetime.now().strftime("%H-%M-%S")

    filename = date_now + "_pyimq_{type}_frc_results.csv".format(type=args.mode)
    filename = os.path.join(output_dir, filename)

    # Get image file names, sort in alphabetic order and complete.
    files_list = list(i for i in os.listdir(path)
                      if i.endswith((".jpg", ".tif", ".tiff", ".png")))
    files_list.sort()
    print(('Number of images to analyze: ', len(files_list)))

    df_main = pandas.DataFrame()

    if args.mode == "regular":
        """
        In the regular mode the images are loaded two at a time. The expectation
        is that the filename sorting above is able to put the images in pairwise
        order. No specific checks are done here to keep things simple.
        """
        def pairwise(iterable):
            a = iter(iterable)
            return zip(a, a)

        for im1, im2 in pairwise(files_list):
            # Read images
            image1 = myimage.MyImage.get_generic_image(os.path.join(path, im1))
            image2 = myimage.MyImage.get_generic_image(os.path.join(path, im2))

            image1.crop_to_rectangle()
            image2.crop_to_rectangle()

            if args.hanning:
                image1.apply_hanning()
                image2.apply_hanning()

            frc_task = frcmeas.FRC(image1, image2, args)
            results = frc_task.get_all()
            npoints = len(results['resolution']['y'])

            title = strutils.common_start(im1, im2)
            title = pandas.Series([title]*npoints, dtype='category')

            df_temp = pandas.DataFrame()
            df_temp['image'] = title
            df_temp['resolution'] = results['resolution']['y']
            df_temp['frcfit'] = results['resolution']['fit']
            df_temp['freq'] = results['resolution']['x']

            df_temp['twosigma'] = results['twosigma']['y']
            df_temp['twosigmafit'] = results['twosigma']['fit']

            df_temp['resolution'] = [results['resolution']['resolution']]*npoints
            df_temp['resolution_x'] = [results['resolution']['x']]*npoints
            df_temp['resolution_y'] = [results['resolution']['y']]*npoints

            df_main = pandas.concat([df_main, df_temp])

    elif args.mode == "single":
        for im in files_list:
            image = myimage.MyImage.get_generic_image(os.path.join(path, im))
            image.crop_to_rectangle()

            if args.hanning:
                image.apply_hanning()

            data = image.get_array()
            odd_index = np.arange(1, data.shape[0], 2)
            even_index = np.arange(0, data.shape[0], 2)
            sub_im1 = data[odd_index, :][:, odd_index]
            sub_im2 = data[even_index, :][:, even_index]

            image1 = myimage.MyImage(sub_im1, image.get_spacing())
            image2 = myimage.MyImage(sub_im2, image.get_spacing())

            frc_task = frcmeas.FRC(image1, image2, args)
            results = frc_task.get_all()
            npoints = len(results['resolution']['y'])

            title = im.split('.')[0]
            df_temp = pandas.DataFrame()
            df_temp['image'] = title
            df_temp['resolution'] = results['resolution']['y']
            df_temp['frcfit'] = results['resolution']['fit']
            df_temp['freq'] = results['resolution']['x']

            df_temp['twosigma'] = results['twosigma']['y']
            df_temp['twosigmafit'] = results['twosigma']['fit']

            df_temp['resolution'] = [results['resolution']['resolution']] * npoints
            df_temp['resolution_x'] = [results['resolution']['x']] * npoints
            df_temp['resolution_y'] = [results['resolution']['y']] * npoints

            df_main = pandas.concat([df_main, df_temp])

    else:
        raise NotImplementedError()

    df_main.index = list(range(len(df_main)))
    df_main.to_csv(filename)

if __name__ == '__main__':
    main()
