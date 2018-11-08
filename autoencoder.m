x = [x Activation(0) * ones(size(x,1),1)];
dim=size(x,2);
w1=rand(dim,hidd);
h=Activation(x*w1);
h=mapminmax(h',0,1)';
h = zscore(h')';
minl=min(h(:));
if minl<0
    h=h-minl;
end
fprintf(1,'Encoder %f: Max Val of Output %f Min Val %f\n',nowlayer,max(h(:)),min(h(:)));

w2=sparse_elm_autoencoder(h,x,1e-3,150)';

[out,w2]=adjustminmax(x,w2);
fprintf(1,'Layer %f: Max Val of Output %f Min Val %f\n',nowlayer,max(out(:)),min(out(:)));

out=Activation(out);
clear w1 h;