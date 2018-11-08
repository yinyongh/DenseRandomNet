function [y]=Activation(x)
global in_p_n;
in_p_n1=in_p_n;
len1=size(x,1);
len2=size(x,2);
y=zeros(len1,len2);
for i=1:len1
    for j=1:len2
        y(i,j)=root(x(i,j),in_p_n1);
    end
end
end

function [y]=root(x,in_p_n)

rc=0.001;
in_p=in_p_n;
in_n=in_p_n;
n=20;
pc=0.05;

ac=pc*(n-1)*(in_n+x);
bc=(- in_p - rc)*n*pc + (- in_n - x)*n + (in_p + rc)*pc - rc;
dc=n*(in_p);

y=(-bc-sqrt(bc*bc-4*ac*dc))/(2*ac);
end
