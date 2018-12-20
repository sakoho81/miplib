% Sami Koho (2015) - Laboratory of Biophysics (University of Turku)
% Description: A simple script for registering two images in a semi-
% automatic manner, by taking advantage of the Matlab Control Point
% Selection tool

% Open images
fixed_filename = uigetfile('*.tif', 'Specify fixed image');
moving_filename = uigetfile('*.tif', 'Specify moving image');

fixed = imread(fixed_filename);
moving = imread(moving_filename);

% Register images using control points
[moving_points, fixed_points] = cpselect(moving, fixed,'Wait', true);
tform = cp2tform(moving_points, fixed_points, 'nonreflective similarity');
fixed_info = imfinfo(fixed_filename);
registered = imtransform(moving, tform,'XData',[1 fixed_info.Width],...
                        'YData',[1 fixed_info.Height]);

% Create and show result
blue = zeros(size(fixed));
result = cat(3, fixed, registered, blue);
imshow(result)

% Save result
output_filename = uiputfile('*.tif', 'Save registration result');
imwrite(result, output_filename);





