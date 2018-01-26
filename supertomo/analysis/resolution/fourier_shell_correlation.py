
import supertomo.processing.ops_myimage as ops_myimage
import supertomo.data.containers.myimage as myimage
import supertomo.analysis.resolution.fourier_ring_correlation as frc

class FSC_Writer(object):
    def __init__(self, output_dir, output_file_prefix):
        self.output_dir = output_dir
        self.output_prefix = output_file_prefix

    def write_table(self, data):
        filename = self.output_prefix + 'globalFSC.csv'
        full_path = os.path.join(self.output_dir, filename)


class DirectionalFSC(frc.FRC):
    def __init__(self, image1, image2, args):

        if image1.get_dimensions() != image2.get_dimensions():
            raise ValueError("The image dimensions do not match")

        if image1.ndim() != 3 or image1.get_dimensions()[0] <= 1:
            raise ValueError("You should provide a stack for FSC analysis")

        #Zoom to isotropic spacing if necessary
        ops_myimage.zoom_to_isotropic_spacing(image1)
        ops_myimage.zoom_to_isotropic_spacing(image2)

        #Call super-class initializer
        super(DirectionalFSC, self).__init__(image1, image2, args)

    @staticmethod
    def find_directional_points_interval(distance_map, start, stop, angle):
        bin_points = find_points_interval(distance_map, start, stop, angle)


#
#         d_theta_rad = d_theta * np.pi / 180.0
#         Thresh = np.cos(d_theta_rad)
#
#         # FFT transforms of the input images
#         self.fft_image1 = np.fft.fftshift(np.fft.fftn(image1.get_array()))
#         self.fft_image2 = np.fft.fftshift(np.fft.fftn(image2.get_array()))
#
#         CosHangle = np.cos(np.angle(self.fft_image1 * np.conj(self.fft_image2)))
#
#
#         startTime = time.time()
#
#         d1 = FFTArray2Real(nx, ny, nz, self.fft_image1);
#         d2 = FFTArray2Real(nx, ny, nz, self.fft_image2);
#         dcH = FFTArray2Real(nx, ny, nz, CosHangle);
#         dFPower = FFTArray2Real(nx, ny, nz, self.fft_image1 * np.conj(self.fft_image1));
#         # dPR = FFTArray2Real(nx,ny,nz,HAngle);
#
#         deltaTime = time.time() - startTime;
#         print("FFTArray2Real performed in %f seconds for size nx=%g " % (deltaTime, nx))
#
#         # FFTArray2Real performed in 22.700670 seconds for size nx=256 , wo jit
#         # FFTArray2Real performed in 0.9	   seconds for size nx=256 , w jit
#         #
#
#         # d1[15][16][17]  is d1.get_value_at(15,16,17)
#         # But d1[15][16][17], for EMData objects is complex...
#         # f[16,17,18] = f[16][17][18]
#         # But this is EMAN's 18,17,16
#
#         # %%		 Section 4a Create FSC Outputs; n1 and n2 are the normalizations of
#         #		  f and g. cH means the cosine of the phase residual.
#
#         nx2 = nx / 2;
#         ny2 = ny / 2;
#         nz2 = nz / 2;
#
#         lsd2 = nx + 2;
#
#         dx2 = 1.0 / float(nx2) / float(nx2);
#         dy2 = 1.0 / float(ny2) / float(ny2);
#         dz2 = 1.0 / float(nz2) / float(nz2);
#         # int inc = Util::round(float(std::max(std::max(nx2,ny2),nz2))/w);
#         w = 1;
#
#         inc = max(nx2, ny2, nz2) / w;
#         inc = int(inc)
#
#         startTime = time.time()
#
#         [retcHGlobal, lr] = CreateFTLikeOutputs(inc, nx, ny, nz, dcH, nx2, ny2, nz2, dx2, dy2, dz2)
#         [ret, n1, n2, lr] = CreateFSCOutputs(inc, nx, ny, nz, d1, d2, nx2, ny2, nz2, dx2, dy2, dz2);
#         [FPower, lr] = CreateFTLikeOutputs(inc, nx, ny, nz, dFPower, nx2, ny2, nz2, dx2, dy2, dz2)
#
#         deltaTime = time.time() - startTime;
#         print("CreateFSCOutputs performed in %f seconds for size nx=%g " % (deltaTime, nx))
#
#         # CreateFSCOutputs performed in	 0.393	seconds for size nx=256 ; using jit
#         # CreateFSCOutputs performed in 45.944	seconds for size nx=256 ; without jit
#         # list(ret)
#
#         # %%		 Section 4b	 Write out FSCs. Define RMax based on this
#
#         # for nx =32, linc is 17
#
#         linc = 0;
#         for i in range(inc + 1):
#             if (lr[i] > 0):
#                 linc += 1;
#
#         result = [0 for i in range(3 * linc)];
#
#         ii = -1;
#         for i in range(inc + 1):
#             if (lr[i] > 0):
#                 ii += 1;
#                 result[ii] = float(i) / float(2 * inc);
#                 result[ii + linc] = float(ret[i] / (np.sqrt(n1[i] * n2[i])));
#                 result[ii + 2 * linc] = lr[i];  # Number of Values
#
#         NormalizedFreq = result[0:(inc + 1)];
#         resultAve = result[(inc + 1):(2 * (inc + 1))];  # This takes values inc+1 values from inc+1 to 2*inc+1
#
#         with open(resultAveOut, "w") as fL1:
#             AveWriter = csv.writer(fL1)
#             for j in range(inc + 1):
#                 valFreqNormalized = NormalizedFreq[j];
#                 valFreq = valFreqNormalized / APixels;
#                 valFSCshell = resultAve[j];
#                 AveWriter.writerow([valFreqNormalized, valFreq, valFSCshell])
#
#         # k0	 = np.array(range(Nxf))/APixels/2.0/Nxf;
#
#         aa = np.abs(np.array(resultAve)) < .13
#         bb = np.where(aa)[0];
#         try:
#             RMax = bb[0] + 4
#         except:
#             RMax = inc
#
#         RMax = min(RMax, inc)
#         print('simple FSC written out to ' + resultAveOut)
#         print('RMax = %d' % (RMax))
#
#         # RMax=40;#	  Change Me	  PRB
#
#         # with open(resultAveOut, "w") as fL1:
#         #	 AveWriter = csv.writer(fL1)
#         #	 for val in resultAve:
#         #		 AveWriter.writerow([val])
#
#         #  Variables
#
#         # result contains normalized freq,	FSC , and number of data points
#         # n1 is the normalization factor for the first	half map over the sphere
#         # n2 is the normalization factor for the second half map over the sphere
#         # ret is the inner product over the whole sphere; becomes inner product after norm
#
#         # &&&&&&&&&&&&&&&&&&&&&&&7		  END PAWEL'S CODE and section 1
#         # %%
#
#         pltName = 'log rot ave FT';
#         ff = plt.figure(pltName);
#         # f.suptitle('Color as Function of Length')
#         Nxf = np.int(nx2) + 1
#         k0 = np.array(range(Nxf)) / APixels / 2.0 / Nxf;
#         k = np.arange(Nxf)
#         fig, ax = plt.subplots()
#         ax.plot(k0, np.log(FPower), 'b', label='FPower')
#         ax.set_xlabel('Spatial Frequency (1/A) ')
#         ax.set_ylabel('log rot ave FT Power')
#
#         legend = ax.legend(loc='upper right', shadow=True)
#
#         fig.savefig(FTOut + '.jpg');
#
#         pltName = 'DPR and global FSC';
#         ff = plt.figure(pltName);
#         # f.suptitle('Color as Function of Length')
#         Nxf = np.int(nx2) + 1
#         k = np.array(range(Nxf)) / 1.07 / 2.0;
#         k = np.array(range(Nxf))
#         fig, ax = plt.subplots()
#         ax.plot(k0, 2 * retcHGlobal / lr, 'b', label='ave cos phase')
#         ax.plot(k0, resultAve, 'g', label='FSC')
#         legend = ax.legend(loc='upper right', shadow=True)
#         ax.set_xlabel('Spatial Frequency (1/A) ')
#         ax.set_ylabel('various FSCs')
#
#         # %%		   Section 5. Create generalized FSC  and FT arrays
#
#         startTime = time.time()
#
#         # The Number at each R is LastInd_OfR+1
#
#         [kXofR, kYofR, kZofR, retofRR, retofRI, n1ofR, n2ofR, NumAtEachR] = \
#             createFSCarrays(nx, ny, nz, lsd2, lr, inc, dx2, dy2, dz2, d1, d2, nx2, ny2, nz2)
#
#         [kXofR, kYofR, kZofR, retcH, retFT, n12ofR] = \
#             createFTarrays(nx, ny, nz, lsd2, lr, inc, dx2, dy2, dz2, dcH, dFPower, nx2, ny2, nz2)
#
#         deltaTime = time.time() - startTime;
#
#         print("FSC arrays created in %f seconds for size nx=%g " % (deltaTime, nx))
#
#         NumAtEachRMax = NumAtEachR[-1]
#
#         kXofR = kXofR[:, :NumAtEachRMax];
#         kYofR = kYofR[:, :NumAtEachRMax];
#         kZofR = kZofR[:, :NumAtEachRMax];
#         retofRR = retofRR[:, :NumAtEachRMax];
#         retofRI = retofRI[:, :NumAtEachRMax];
#         n1ofR = n1ofR[:, :NumAtEachRMax];
#         n2ofR = n2ofR[:, :NumAtEachRMax];
#
#         NumAtEachRMax = NumAtEachR[RMax];
#         NumAtEachRMaxCuda = 15871;  # NumAtEachR[50];#15871
#
#         # MaxLoopsIllNeed = NumAtEachRMax*NumAtEachRMax/NumAtEachRMaxCuda/NumAtEachRMaxCuda;
#
#         # kXofR,kYofR, kZofR
#         # retofRR,retofRI
#         # n1ofR,n2ofR
#         # NumAtEachR is a one d arry indicating	 the unique number of sites at each radius
#
#         # Some Tests  j=2;list(kXofR[j][0:NumAtEachR[j]])
#         # j=2;list(retofRI[j][0:NumAtEachR[j]])
#         # j=2;list(retofRR[j][0:NumAtEachR[j]])
#         # j=2;list(n1ofR[j][0:NumAtEachR[j]])
#         # j=2;list(n2ofR[j][0:NumAtEachR[j]])
#
#         # FSC arrays created in 1.399719 seconds for size nx=128 using jit (vs 5 seconds)
#         # FSC arrays created in 2		 seconds for size nx=256 using autjit
#
#         # If one normally indexes a square  array as (X,Y)
#         # then the upper right part would have index
#         #	   N(X-1) - (X)(X-1)/2 + Y
#         # The greatest Element would be when X=Y=N
#         #	 N(N-1)/2 +N = N(N+1)/2
#         # For N=1; gives 1. For N=2,
#
#         # vv=NumAtEachRMax;
#         # hh= int(vv*(vv+1)/2)
#         # hh=5.1972* pow(10,9)
#
#         # %%	   Section 6. Average on Shells
#
#         if 0:
#             RMax = 30;
#             RMax = 81;
#             RMax = 47;
#             RMax = 60;
#
#         startTime = time.time()
#
#         [retofROutR, retofROutI, n1ofROut, n2ofROut, NumAtROut] = \
#             AveragesOnShellsUsingLogicB(inc, retofRR, retofRI, n1ofR, n2ofR, kXofR, kYofR, kZofR, \
#                                         NumAtEachR, Thresh, RMax);
#
#         if 0:
#             [retofRcH, retofRFT, n12ofROut, n21ofROut, NumAtROut] = \
#                 AveragesOnShellsUsingLogicB(inc, retcH, retFT, n12ofR, n12ofR, kXofR, kYofR, kZofR, \
#                                             NumAtEachR, Thresh, RMax);
#
#         # j=2; list(retofROutR[j][:NumAtEachR[j]]) perfect
#         # list(retofROutI[2][:NumAtEachR[2]]) perfect
#         #  list(n1ofROut[2][:NumAtEachR[2]])  perfect
#         #  list(n2ofROut[2][:NumAtEachR[2]]) perfect
#         #  list(NumAtROut[2][:NumAtEachR[2]]) perfect
#
#         deltaTime = time.time() - startTime;
#
#         print("AveragesOnShells created in %f seconds for size nx=%g " % (deltaTime, nx))
#
#         # no jit AveragesOnShells created in 134.065687 seconds for size nx=64
#         #  jit AveragesOnShells created in 58.065687 seconds for size nx=64
#         # jit AveragesOnShells created in 0.232876 seconds for size nx=32
#         # jit on inner loop (or both loops) AveragesOnShells created in 2.050189 seconds for size nx=64
#         # jit AveragesOnShells created in 26.903263 seconds for size nx=128
#         # cudajit Average On shells created in 14.11 seconds for size nx=256
#         # cudajit AveragesOnShells created in 0.058333 seconds for size nx=32
#         # cudajit AveragesOnShells created in 2.212252 seconds for size nx=256
#         # AveragesOnShells created in 1602.677335 seconds for size nx=128  without matrix multiply
#         # AveragesOnShells created in 149  seconds for size nx=128	with matrix multiply
#         #
#         #  NumAtROutPre.shape 1311 1311
#         # sum, sum of NumAtROutPre = 104763
#         # r =15, jLoop = 0
#         # NumAtROutPre created in 2.635284 seconds, retofROutRPre  in 0.044771 seconds for size r=15
#
#         # NumAtROutPre created in 87.902457 seconds for size r=128
#         # retofROutRPre created in 19.214520 seconds for size r=128
#         # AveragesOnShells created in 107.825461 seconds for size nx=256
#
#         if 0:
#             h5f_write = h5py.File(output_dir + 'Radial' + 'ResEM' + OutputStringLabel + 'Out.hdf', 'w')
#             h5f_write.create_dataset('MDF/images/0/image', data=retofROutR)
#             # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#             h5f_write.close()
#
#             h5f_write = h5py.File(output_dir + 'Radial' + 'cH' + OutputStringLabel + 'Out.hdf', 'w')
#             h5f_write.create_dataset('MDF/images/0/image', data=retofRcH)
#             # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#             h5f_write.close()
#
#             h5f_write = h5py.File(output_dir + 'Radial' + 'FT' + OutputStringLabel + 'Out.hdf', 'w')
#             h5f_write.create_dataset('MDF/images/0/image', data=retofRFT)
#             # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#             h5f_write.close()
#
#             h5f_write = h5py.File(output_dir + 'Radial' + 'n12' + OutputStringLabel + 'Out.hdf', 'w')
#             h5f_write.create_dataset('MDF/images/0/image', data=n12ofROut)
#             # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#             h5f_write.close()
#
#             h5f_write = h5py.File(output_dir + 'Radial' + 'n1' + OutputStringLabel + 'Out.hdf', 'w')
#             h5f_write.create_dataset('MDF/images/0/image', data=n1ofROut)
#             # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#             h5f_write.close()
#
#             h5f_write = h5py.File(output_dir + 'Radial' + 'n2' + OutputStringLabel + 'Out.hdf', 'w')
#             h5f_write.create_dataset('MDF/images/0/image', data=n2ofROut)
#             # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#             h5f_write.close()
#
#         if 0:
#             fig = plt.figure()
#             ax = fig.gca()
#             # ThisHist = ax.hist(GuessedA*20)
#             j = 47;
#             ax.scatter(retFT[j, :NumAtEachR[j]] / n12ofROut[j, :NumAtEachR[j]],
#                        retcH[j, :NumAtEachR[j]] / n12ofROut[j, :NumAtEachR[j]]);
#             ax.scatter(retFT[j, :NumAtEachR[j]], retcH[j, :NumAtEachR[j]]);
#             ax.set_xlabel('FT power');
#             ax.set_ylabel('Phase Residual');
#             ax.set_title('Phase Residual vs FT Power');
#         # fig.savefig('PointWeightsTimesValues1D.jpg')
#
#         # %%	  Section 7. We have the unaveraged quantities
#
#         startTime = time.time()
#         [ResEMR, ResEMI, ResNum, ResDen, ResultR, ResultI] = \
#             NormalizeShells(nx, ny, nz, kXofR, kYofR, kZofR, inc, retofROutR, retofROutI, n1ofROut, n2ofROut, NumAtEachR,
#                             RMax);
#         deltaTime = time.time() - startTime;
#
#         print("NormalizeShells created in %f seconds for size nx=%g, RMax=%g " % (deltaTime, nx, RMax))
#
#         # NormalizeShells created in 3.175528 seconds for size nx=64; wo jit
#         # NormalizeShells created in 3.175528 seconds for size nx=64
#         # NormalizeShells created in 3.602324 seconds for size nx=32
#         # NormalizeShells created in 0.401217 seconds for size nx=32
#         # NormalizeShells created in 3.374886 seconds for size nx=256 ; wo jit
#         # NormalizeShells created in infinity  seconds for size nx=256 ; w jit
#         # list(ResEMR[65,65,:]) for GS IR protein
#         # list(ResEMR[63,63,:]) for HA Sh2 protein
#
#         # %%
#         #	 Section 8.		 Write Out FSC volumes to file
#         #
#
#         # csvfile=open(OutP1csvFN,'w')
#         # P1writer= csv.writer(csvfile,delimiter=' ',quotechar='|');
#         # P1writer.writerow(PolynomialL1);
#         #
#         #
#         # if 0:
#         #	 ResEMRB=np.zeros([334,334,334])
#         #	 ResEMRB[:,:,:] = ResEMR[:334,:334,:334]
#         #	 mrc.write(ResEMR,'Proteasome_FSC.mrc')
#         #	 ResEMR.write_mrc('Proteasome_FSC.mrc');
#         #	 ResNum.write_mrc('Proteasome_Num.mrc');
#         #	 ResDen.write_mrc('Proteasome_Den.mrc');
#         #	 h5f_write = h5py.File('IR_FSC.hdf','w')
#         #	 h5f_write.create_dataset('MDF/images/0/image',data=ResEMR)
#         #	 # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#         #	 h5f_write.close()
#         #	 #
#         #	 h5f_write = h5py.File('Proteasome_FSC.hdf','w')
#         #	 h5f_write.create_dataset('MDF/images/0/image',data=ResEMR)
#         #	 # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#         #	 h5f_write.close()
#         #	 #
#         #	 h5f_write = h5py.File('Proteasome_HalfMap1.hdf','w')
#         #	 h5f_write.create_dataset('MDF/images/0/image',data=f)
#         #	 # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#         #	 h5f_write.close()
#         #
#
#         if 0:
#             h5f_write = h5py.File('ResEMOutHDF_FN.hdf', 'w')
#             h5f_write.create_dataset('MDF/images/0/image', data=ResEMR.T)
#             # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#             h5f_write.close()
#
#         # print('ResEMR'); print(type(ResEMR))
#
#         if 1:
#             ResEMRT = ResEMR.T;
#             mrc_write = mrcfile.new(ResEMOutHDF_FN, overwrite=True)
#             mrc_write.set_data(ResEMRT.astype('<f4'))
#             mrc_write.voxel_size = (float(APixels), float(APixels), float(APixels))
#             mrc_write.update_header_from_data()
#             mrc_write.close()
#
#         # %%
#
#         if 0:  # Legacy code that includes axes for plotting
#             ResEMRPlus = AddAxes(ResEMR, 2, 10)
#             h5f_write = h5py.File('ResEMROutWithAxes.hdf', 'w')
#             h5f_write.create_dataset('MDF/images/0/image', data=ResEMRPlus.T)
#             # <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#             h5f_write.close()
#
#         # %%
#         #	 Section 9;		  Write Out		 PolynomialL1, PolynomialL2, N2, resultAve to file
#
#         if 0:
#             startTime = time.time()
#             [PolynomialL1, PolynomialL2] = ReturnL1L2Moments(inc, LastInd_OfR, kXofR, kYofR, kZofR, n1ofR, n2ofR, ResultR,
#                                                              ResultI);
#             deltaTime = time.time() - startTime;
#             print("ReturnL1L2Moments created in %f seconds for size nx=%g" % (deltaTime, nx))
#
#         # ReturnL1L2Moments created in 0.13 seconds for size nx=32; jit 1.81 seconds
#
#         # %%	  Section 10 Plot 5 Axes
#         [xf, yf, zf] = ExtractAxes(ResEMR);
#         Nxf = len(xf)
#         DPRf = 2 * retcHGlobal[:-1] / lr[:-1];
#         Globalf = resultAve[:-1];
#
#         # pltName='ProteosomePlots';
#         # pltName='HAPlots';
#         # pltName='IRPlots';
#         pltName = PlotsOut;
#         ff = plt.figure(pltName);
#         # f.suptitle('Color as Function of Length')
#
#         k = np.array(range(Nxf)) / 1.07 / 2.0;
#         k0 = np.array(range(Nxf)) / APixels / 2.0 / Nxf;
#
#         k = np.array(range(Nxf))
#         fig, ax = plt.subplots()
#         ax.plot(k0, xf, 'b', label='x dir')
#         ax.plot(k0, yf, 'g', label='y dir')
#         ax.plot(k0, zf, 'r', label='z dir')
#         ax.plot(k0, DPRf, 'k', label='ave cos phase')
#         ax.plot(k0, Globalf, 'y', label='global FSC')
#
#         ax.set_xlabel('Spatial Frequency (1/A) ')
#         ax.set_ylabel('FSCs')
#
#         # Now add the legend with some customizations.
#         legend = ax.legend(loc='upper right', shadow=True)
#
#         fig.savefig(pltName + '.jpg');
#
#         # ResEMR[:,ny2,nz2]
#
#         xyzf = np.reshape(np.concatenate((xf, yf, zf, DPRf, Globalf)), (5, Nxf))
#         xyzf = xyzf.T;
#         np.savetxt(PlotsOut + '.csv', xyzf)
#
#         # %%
#         # sys.exit()
#
#         ## Flush out plots
#         plt.clf()
#         plt.cla()
#         plt.close()
#
#         # %%
#
#         # fOut = AddAxes(f,0);
#         #
#         #
#         # h5f_write = h5py.File('Proteasome_withz.hdf','w')
#         # h5f_write.create_dataset('MDF/images/0/image',data=fOut)
#         ## <HDF5 dataset "array": shape (63, 63, 63), type "<f8">
#         # h5f_write.close()
#         #
#         #
#         #
#         enablePrint()
#
#
# if __name__ == "__main__":
#     main(argv[1], argv[2], argv[3], float(argv[4]), float(argv[5]))
