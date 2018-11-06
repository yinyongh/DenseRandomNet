% This is an example of applying the dense random neural network
% proposed in [1-3] to classify the small Norb dataset from [4].
% Please cite the following papers if you use this code.
% [1] Gelenbe, Erol; Yin, Yonghua. Deep Learning with Random Neural Networks. 2016 International Joint Conference on
% Neural Networks (IJCNN) 2016, pp. 1633–1638.
% [2] Yin, Yonghua; Gelenbe, Erol. Deep learning in multi-layer architectures of dense nuclei.
% arXiv preprint arXiv:1609.07160 2016.
% [3] Gelenbe, Erol; Yin, Yonghua. Deep Learning with Dense Random Neural Networks. International Conference on 
% Man-Machine Interactions. Springer, 2017, pp. 3-18.
% [4] Y. LeCun, F.J. Huang, L. Bottou, Learning Methods for Generic Object Recognition with Invariance to Pose and Lighting. 
% IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR) 2004

clear all;

DataFile='smallnorb-32x32.mat';
in_p_n=0.005;
neurnum={500 2000};
channelnum=2;
[o,TargetValue,vo,VTargetValue,TrainingTime,TestingTime,TrainingAccuracy,TestingAccuracy]=MCMLDRNN(DataFile,in_p_n,neurnum,channelnum);
TrainingAccuracy
TestingAccuracy
name=strcat('MCMLDRNN_for_norb.mat');
save(name,'TrainingAccuracy','TestingAccuracy','TrainingTime','TestingTime');

