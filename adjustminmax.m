function [h,w]=adjustminmax(x,w)
    h=x*w;
    l=max(h(:));
    h=h/max(l,eps);
    w=w/max(l,eps);
    
    h=h/10;
    w=w/10;
end