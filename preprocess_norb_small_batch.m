% This is the code for loading and structuring the small NORB dataset downloaded
% from https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/ [1].
% [1] Y. LeCun, F.J. Huang, L. Bottou, Learning Methods for Generic Object Recognition with Invariance to Pose and Lighting. 
% IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR) 2004

%% load and structure the training dataset
% Note that all images in the dataset are downsampled from 96x96 images into
% 32x32 images.

close all;
clear all;
fid=fopen('norb_dataset\smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','r');
fread(fid,4,'uchar');   % result = [85 76 61 30], byte matrix(in base 16: [55 4C 3D 1E])
fread(fid,4,'uchar');   % result = [4 0 0 0], ndim = 4
fread(fid,4,'uchar');   % result = [236 94 0 0], dim0 = 24300 (=94*256+236)
fread(fid,4,'uchar');   % result = [2 0 0 0],     dim1 = 2
fread(fid,4,'uchar');   % result = [96 0 0 0],   dim2 = 96
fread(fid,4,'uchar');   % result = [96 0 0 0],   dim3 = 96

% create a mask to downsample the 96x96 images into 32x32 images
mask_image = ones(96,96);
for i = 1:3:96
    mask_image(i:i+1,:) = 0;
end
for i = 1:3:96
    mask_image(:,i:i+1) = 0;
end
mask_image = reshape(mask_image, 1, 96*96);
mask_image = [mask_image mask_image];
mask_image = find(mask_image == 1);

% read the training data batch by batch: each sample contains two 96x96 images
train_x = [];
num = [5000 5000 5000 5000 4300];
for i = 1:length(num)
    train_x_tmp = transpose(reshape(fread(fid,96*96*2*num(i)),96*96*2,num(i)));
    % use the mask to downsample the 96x96 images into 32x32 images
    train_x = [train_x; train_x_tmp(:,mask_image)];
end

% show a sample of the 96x96 image in Figure 1
figure(1);
imshow(reshape(train_x_tmp(end,1:96*96), 96,96),[0 255]);
% show a sample of the 32x32 image in Figure 2 after downsampling
figure(2);
imshow(reshape(train_x(end,1:32*32), 32,32),[0 255]);
clear train_x_tmp;

% read all the training labels
fid=fopen('norb_dataset\smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat','r');
fread(fid,4,'uchar');   % result = [84 76 61 30], int matrix (54 4C 3D 1E)
fread(fid,4,'uchar');   % result = [1 0 0 0],   ndim = 1
fread(fid,4,'uchar');   % result = [236 94 0 0],  dim0 = 24300
fread(fid,4,'uchar');   % result = [1 0 0 0]  (ignore this integer)
fread(fid,4,'uchar');   % result = [1 0 0 0]  (ignore this integer)
trainlabels = fread(fid,24300,'int');    % result = [0 1 2 3 4 0 1 2 3 4 ... ]
label=double(trainlabels+1);

% combine the training samples and labels together for training
TrainingData=[train_x,label];

%% load and structure the testing dataset
fid=fopen('norb_dataset\smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','r');
fread(fid,4,'uchar');  
fread(fid,4,'uchar'); 
fread(fid,4,'uchar'); 
fread(fid,4,'uchar');   
fread(fid,4,'uchar');  
fread(fid,4,'uchar');


% read the testing data batch by batch: each sample contains two 96x96 images
test_x = [];
num = [5000 5000 5000 5000 4300];
for i = 1:length(num)
    test_x_tmp = transpose(reshape(fread(fid,96*96*2*num(i)),96*96*2,num(i)));
    % use the mask to downsample the 96x96 images into 32x32 images
    test_x = [test_x; test_x_tmp(:,mask_image)];
end

% show a sample of the 96x96 image in Figure 1
figure(3);
imshow(reshape(test_x_tmp(end,1:96*96), 96,96),[0 255]);
% show a sample of the 32x32 image in Figure 2 after downsampling
figure(4);
imshow(reshape(test_x(end,1:32*32), 32,32),[0 255]);
clear test_x_tmp;

% read all the testing labels
fid=fopen('norb_dataset\smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat','r');
fread(fid,4,'uchar');   % result = [84 76 61 30], int matrix (54 4C 3D 1E)
fread(fid,4,'uchar');   % result = [1 0 0 0],   ndim = 1
fread(fid,4,'uchar');   % result = [236 94 0 0],  dim0 = 24300
fread(fid,4,'uchar');   % result = [1 0 0 0]  (ignore this integer)
fread(fid,4,'uchar');   % result = [1 0 0 0]  (ignore this integer)
testlabels = fread(fid,24300,'int');    % result = [0 1 2 3 4 0 1 2 3 4 ... ]
Vlabel=double(testlabels+1);

% combine the testing samples and labels together
TestingData=[test_x,Vlabel];

%% save the structed datasets for future use
save('smallnorb-32x32.mat', 'TrainingData','TestingData');
clear all;