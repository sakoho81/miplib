#!/usr/bin/env python
# -*- coding: UTF-8 -*-
### Require Anaconda3

### ============================
### 3D FSC Software Wrapper
### Written by Philip Baldwin
### Edited by Yong Zi Tan and Dmitry Lyumkis
### Anaconda environment by Carl Negro
### Downloaded from https://github.com/nysbc/Anisotropy
### git clone https://github.com/nysbc/Anisotropy.git
### 
### See Paper:
### Addressing preferred specimen orientation in single-particle cryo-EM through tilting
### 10.1038/nmeth.4347
###
### Credits:
### 1) UCSF Chimera, especially Tom Goddard
### 2) mrcfile 1.0.0 by Colin Palmer (https://github.com/ccpem/mrcfile)
###
### Version 2.5 (5 September 2017)
### 
### Revisions 
### 1.1 - Added mpl.use('Agg') to allow matplotlib to be used without X-server
###     - Added Sum of Standard Deviation
### 1.2 - Added FSCCutoff Option
### 2.0 - Incorporation of AutoJIT version of 3D FSC for 10x faster processing
### 2.1 - 3D FSC takes MRC files
### 2.2 - Fixed bugs with 3DFSC missing a line in volume, and plotting title error
### 2.3 - Fixed various bugs, new thresholding algorithm, added progress bar, improved Chimera plotting, more error checking
### 2.4 - Incorporation with website, use Click package
### 2.5 - Outputs raw histogram data
### ============================

import os
import sys

import datetime
import numpy as np
import time
import supertomo.ui.utils as uiutils

import supertomo.analysis.resolution.fourier_shell_correlation as fsc
import ThreeDFSC_Analysis # Version 5.0 Latest

import supertomo.ui.resolution_options as resops
import supertomo.ui.utils as uiutils
import supertomo.data.io.read as imread

# Check Anaconda version


def main():

    options = resops.get_fsc_script_options(sys.argv[1:])

    # Convert file paths to absolutes
    fullmap_path = uiutils.get_full_path(options.fullmap, options.dir)
    halfmap1_path = uiutils.get_full_path(options.halfmap1, options.dir)
    halfmap2_path = uiutils.get_full_path(options.halfmap2, options.dir)
    if halfmap1_path != halfmap2_path:
        raise ValueError("Halfmap1 and Halfmap2 should not be equal.")

    # Open images
    halfmap1 = imread.get_image(halfmap1_path, 'myimage')
    halfmap2 = imread.get_image(halfmap2_path, 'myimage')

    #TODO: This doesn't make any sense. Add an option to crop.
    if halfmap1.get_dimensions() != halfmap2.get_dimensions():
        raise ValueError("The dimensions of Halfmap1 and Halfmap2 should match")

    # Calculate mask if necessary
    if options.mask is not None:
        mask_path = uiutils.get_full_path(options.mask, options.dir)
        mask = imread.get_image(mask_path, 'myimage')
        halfmap1 *= mask
        print ("Masking performed")

    # Check numThresholdsForSphericityCalcs is bigger than 0
    if options.numThresholdsForSphericityCalcs < 0:
        raise ValueError("Please key in a positive integer for the --numThresholdsForSphericityCalcs option.")

    # Create a output directory and a filename template
    output_dir = datetime.datetime.now().strftime("%Y-%m-%d") + '_threedfrc_output'
    output_dir = os.path.join(options.dir, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name_template = datetime.datetime.now().strftime("%H-%M-%S") + '_threedfrc'
    # Part 01
    # Generate FRC map
    print("Step 01: Generating 3DFSC Volume")
    fsc.main(halfmap1, halfmap2, output_dir, file_name_template, options.dthetaInDegrees)

    

    #TODO: Figure out how this thing works.
    os.system("cp Results_" + options.ThreeDFSC + "/ResEM" + options.ThreeDFSC + "Out.mrc Results_" + options.ThreeDFSC + "/" + options.ThreeDFSC + ".mrc")

    print ("3DFSC Results_" + options.ThreeDFSC + "/" + options.ThreeDFSC + ".mrc generated.")


    # Part 02
    print("Step 02: Generating Analysis Files")
    ThreeDFSC_Analysis.main(halfmap1_path,halfmap2_path,fullmap,options.apix,options.ThreeDFSC,options.dthetaInDegrees,options.histogram,options.FSCCutoff,options.ThresholdForSphericity,options.HighPassFilter,options.numThresholdsForSphericityCalcs)
    print ("Done")
    print ("Results are in the folder Results_%s" % output_dir)
    #print ("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()



