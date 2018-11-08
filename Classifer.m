function [o]=Classifer(x,w)
layer=length(w{1});
source=length(x);
for now_source=1:source
    if layer>2
        [o1]=mlrnn(x{now_source},w{now_source}(1:layer-2));
    else
        o1=x{now_source};
    end
    if exist('o','var')
        o=[o o1];
    else
        o=o1;
    end
end
o=o*w{1}{layer-1};
fprintf(1,'Layer %f: Max Val of Output %f Min Val %f\n',layer-1,max(o(:)),min(o(:)));
o=Activation(o);
o=o*w{1}{layer};
end