function [o]=mlrnn(x,w)
    layer=length(w);
    o=x;
    clear x;
    for i=1:layer
        o=o*w{i};
        fprintf(1,'Layer %f: Max Val of Output %f Min Val %f\n',i,max(o(:)),min(o(:)));
        o=Activation(o);
    end
end