function [o,TargetValue,vo,VTargetValue,TrainingTime,TestingTime,TrainingAccuracy,TestingAccuracy,w,max_data_val,min_data_val,in_p_n1]=MCMLDRNN(DataFile,in_p_n1,neurnum,channelnum)
% Please cite the following papers if you use this code.
% [1] Gelenbe, Erol; Yin, Yonghua. Deep Learning with Random Neural Networks. 2016 International Joint Conference on
% Neural Networks (IJCNN) 2016, pp. 1633�1638.
% [2] Yin, Yonghua; Gelenbe, Erol. Deep learning in multi-layer architectures of dense nuclei.
% arXiv preprint arXiv:1609.07160 2016.
% [3] Gelenbe, Erol; Yin, Yonghua. Deep Learning with Dense Random Neural Networks. International Conference on 
% Man-Machine Interactions. Springer, 2017, pp. 3-18.
% [4] Yonghua Yin. Random Neural Network Methods and Deep Learning. Submitted to Sensors, 2018.
%
% Note that certain functions of the source code are built based on the
% source codes shared by [5]. It is also suggested to cite [5] if this code
% is used.
% [5] Tang, Jiexiong; Deng, Chenwei; Huang, Guang-Bin. Extreme Learning Machine for Multilayer Perceptron. 
% IEEE Transactions on Neural Networks and Learning Systems, vol. 27, no. 4, pp. 809-821, April 2016.
% 
% 
% By Yonghua Yin
% Intelligent Systems and Networks Group, 
% Department of Electrical and Electronic Engineering, Imperial College,
% London SW7 2BT, UK
% Personal website: http://www.yonghuayin.icoc.cc
% Emails: yinyongh@foxmail.com; y.yin14@imperial.ac.uk

format long;
neurnum=cell2mat(neurnum);
load(DataFile, 'TrainingData','TestingData');
Dimension=size(TrainingData,2)-1;
train_x=TrainingData(:,1:Dimension);
TargetValue=TrainingData(:,Dimension+1);

test_x=TestingData(:,1:Dimension);
VTargetValue=TestingData(:,Dimension+1);

clear TrainingData;
clear TestingData;

NumSimuPoint=size(train_x,1);
VNumSimuPoint=size(test_x,1);

label=TargetValue;
Vlabel=VTargetValue;

alldata=cat(1,TargetValue,VTargetValue);
class=length(unique(alldata));
clear alldata;

temp=zeros(NumSimuPoint,class);
for i=1:NumSimuPoint
    temp(i,TargetValue(i))=1;
end
TargetValue=temp;

temp=zeros(VNumSimuPoint,class);
for i=1:VNumSimuPoint
    temp(i,VTargetValue(i))=1;
end
VTargetValue=temp;

clear temp;

global in_p_n;
in_p_n=in_p_n1;

train_y=TargetValue;
test_y=VTargetValue;

clear x Vx;

layer=length(neurnum);

min_data_val=min(min(train_x(:)),min(test_x(:)));
if min_data_val<0
    train_x=train_x-min_data_val;
    test_x=test_x-min_data_val;
end
max_data_val=max(max(train_x(:)),max(test_x(:)));
if max_data_val>0
    train_x=train_x/max_data_val;
    test_x=test_x/max_data_val;
end

temp=train_x;
clear train_x;
num=Dimension/channelnum;
for i=1:channelnum
    beg1=(i-1)*num+1;
    end1=i*num;
    train_x{i}=temp(:,beg1:end1);
end
clear temp;

temp=test_x;
clear test_x;
for i=1:channelnum
    beg1=(i-1)*num+1;
    end1=i*num;
    test_x{i}=temp(:,beg1:end1);
end
clear temp;

source=length(train_x);

tic;
%%
%the RNN layers
clear feature1;
for nowsource=1:source
    if layer>1
        for nowlayer=1:layer-1
            if nowlayer==1
                x=train_x{nowsource};
            else
                x=out;
            end
            hidd=neurnum(nowlayer);
            autoencoder;
            w{nowsource}{nowlayer}=w2;
        end
        clear w2;
        if exist('feature1','var')
            feature1=[feature1 out];
        else
            feature1=out;
        end
    else
        if exist('feature1','var')
            feature1=[feature1 train_x{nowsource}];
        else
            feature1=train_x{nowsource};
        end
    end
end

%%
%the ELM layer
x=feature1;
clear out feature;
hidd=neurnum(layer);

x = [x Activation(0) * ones(size(x,1),1)];
dim=size(x,2);
w1=rand(dim,hidd);

[h,w1]=adjustminmax(x,w1);
fprintf(1,'Layer %f: Max Val of Output %f Min Val %f\n',layer,max(h(:)),min(h(:)));

out=Activation(h);

w{1}{layer}=w1;
clear w1;

w{1}{layer+1}=pinv(out)*train_y;

TrainingTime=toc;

o=out*w{1}{layer+1};

[TrainingAccuracy]=top_acc(o,label,1);

%%
%make the weight connection smooth
for now_source=1:source
    if layer>2
        for nowlayer=1:layer-2
            temp=w{now_source}{nowlayer};
            temp=[temp zeros(size(temp,1),1)];
            w{now_source}{nowlayer}=temp;
        end
    end
end

if layer>1
    for nowlayer=layer-1:layer-1
        temp=w{source}{nowlayer};
        temp=[temp zeros(size(temp,1),1)];
        w{source}{nowlayer}=temp;
    end
end


tic;
%%
%testing dataset
for now_source=1:source
    temp=test_x{now_source};
    test_x{now_source} = [temp Activation(0) * ones(size(temp,1),1)];
end
[vo]=Classifer(test_x,w);
TestingTime=toc;
[TestingAccuracy]=top_acc(vo,Vlabel,1);

end
