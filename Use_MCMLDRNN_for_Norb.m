% This is an example of applying the dense random neural network
% proposed in [1-3] to classify the small Norb dataset from [4].
% Please cite the following papers if you use this code.
% [1] Gelenbe, Erol; Yin, Yonghua. Deep Learning with Random Neural Networks. 2016 International Joint Conference on
% Neural Networks (IJCNN) 2016, pp. 1633�1638.
% [2] Yin, Yonghua; Gelenbe, Erol. Deep learning in multi-layer architectures of dense nuclei.
% arXiv preprint arXiv:1609.07160 2016.
% [3] Gelenbe, Erol; Yin, Yonghua. Deep Learning with Dense Random Neural Networks. International Conference on 
% Man-Machine Interactions. Springer, 2017, pp. 3-18.
% [4] Yin, Yonghua. "Deep Learning with the Random Neural Network and its Applications." arXiv preprint arXiv:1810.08653 (2018).
%
%
% By Yonghua Yin
% Intelligent Systems and Networks Group, 
% Department of Electrical and Electronic Engineering, Imperial College,
% London SW7 2BT, UK
% Personal website: http://www.yonghuayin.icoc.cc
% Emails: yinyongh@foxmail.com; y.yin14@imperial.ac.uk

%%
% pre-process dataset
% The dataset is obtained from [5] and downsampled by the code "pre_process_norb.m".
% [5] Y. LeCun, F.J. Huang, L. Bottou, Learning Methods for Generic Object Recognition with Invariance to Pose and Lighting. 
% IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR) 2004
%%
%traing the MCMLDRNN
clear all;
DataFile='smallnorb-32x32.mat';
in_p_n=0.005;
neurnum={150 2000};
channelnum=2;
[o,TargetValue,vo,VTargetValue,TrainingTime,TestingTime,TrainingAccuracy,TestingAccuracy,w,maxl,minl,in_p_n]=MCMLDRNN(DataFile,in_p_n,neurnum,channelnum);
TrainingAccuracy
TestingAccuracy
name=strcat('MCMLDRNN_for_norb.mat');
save(name,'TrainingAccuracy','TestingAccuracy','TrainingTime','TestingTime','w','maxl','minl','in_p_n');
clear all;

%%
%testing a trained MCMLDRNN with testing dataset
clear all;
load('smallnorb-32x32.mat', 'TestingData');
Dimension=size(TestingData,2)-1;
test_x=TestingData(:,1:Dimension);
Vlabel=TestingData(:,Dimension+1);
tic;
name=strcat('MCMLDRNN_for_norb.mat');
global in_p_n;
load(name,'w','maxl','minl','in_p_n');
channelnum = length(w);
if minl<0
    test_x=test_x-minl;
end
if maxl>0
    test_x=test_x/maxl;
end
temp=test_x;
clear test_x;
num=Dimension/channelnum;
for i=1:channelnum
    beg1=(i-1)*num+1;
    end1=i*num;
    test_x{i}=temp(:,beg1:end1);
end
clear temp;

for now_source=1:channelnum
    temp=test_x{now_source};
    test_x{now_source} = [temp Activation(0) * ones(size(temp,1),1)];
end
[vo]=Classifer(test_x,w);
TestingTime=toc;
[TestingAccuracy]=top_acc(vo,Vlabel,1)
