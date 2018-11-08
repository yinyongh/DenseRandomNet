function [acc]=top_acc(y,label,top)
    num=size(y,1);
    right=0;
    for i=1:num
        v=y(i,:);
        b=0;
        for j=1:top
            maxv=max(v);
            minv=min(v);
            index=find(v==maxv);
            if index(1)==label(i)
                b=1;
            end
            v(index(1))=minv;
        end
        right=right+b;
    end
    acc=right/num;
end