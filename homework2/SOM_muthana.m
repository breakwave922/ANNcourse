% self organizing map Kohonen
clc
clear all
close all
sample_dim=2;
sample_size=1000;
N=10;

%%
% % 1- generate rand 
% x=rand(sample_size,sample_dim);

% % 2- draw the circle 
% a = rand(10000, 2);
% c = repmat([0.5 0.5], 10000, 1);
% x = a(find(eucdist(a, c) < 0.5), :);

% 3-draw  ring
a = rand(10000, 2);
c = repmat([0.5 0.5], 10000, 1);
acdist=sum((a-c).^2,2).^0.5;
a = a(find(acdist < 0.5), :);
c = c(1:size(a, 1),:);
acdist=sum((a-c).^2,2).^0.5;
a = a(find(acdist > 0.3), :);
x=a;
%%
w1=rand(N,N);
w2=rand(N,N);
figure(1)
plot(x(:,1),x(:,2),'.b')
hold on
plot(w1,w2,'or')
plot(w1,w2,'k','linewidth',2)
plot(w1',w2','k','linewidth',2)
hold off
title('t=0');
pause(0.05)
ite=600;
t=1;
x1= x(:,1);
x2= x(:,2);
while (t<=ite)
    alpha=(1-(t/ite));           % learning rate 
    d=ceil(5*(1-(t/ite)));       % distance from the excited neuron to the wienner, it will decrease based on the iret.
    %loop for the sample_size inputs
    for i=1:sample_size
        dist=(x1(i)-w1).^2+(x2(i)-w2).^2;
        [minvalue,ind]=min(dist(:));                  % find the winner neuron
        [I,J] = ind2sub([size(dist,1) size(dist,2)],ind); % idices for winner neuron
        j1star= I;
        j2star= J;
        %update the winning neuron
        w1(j1star,j2star)=w1(j1star,j2star)+alpha*(x1(i)- w1(j1star,j2star));
        w2(j1star,j2star)=w2(j1star,j2star)+alpha*(x2(i)- w2(j1star,j2star));
        %update the neighbour neurons
        for dd=1:1:d  
            jj1=j1star-dd;
            jj2=j2star;
            if (jj1>=1)
                w1(jj1,jj2)=w1(jj1,jj2)+alpha*(x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2)+alpha*(x2(i)-w2(jj1,jj2));
            end
            jj1=j1star+dd;
            jj2=j2star;
            if (jj1<=N)
                w1(jj1,jj2)=w1(jj1,jj2)+alpha*(x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2)+alpha*(x2(i)-w2(jj1,jj2));
            end
            jj1=j1star;
            jj2=j2star-dd;
            if (jj2>=1)
                w1(jj1,jj2)=w1(jj1,jj2)+alpha*(x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2)+alpha*(x2(i)-w2(jj1,jj2));
            end
            jj1=j1star;
            jj2=j2star+dd;
            if (jj2<=N)
                w1(jj1,jj2)=w1(jj1,jj2)+alpha*(x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2)+alpha*(x2(i)-w2(jj1,jj2));
            end
        end
    end
    t=t+1;
    figure(1)
    plot(x1,x2,'.b')
    hold on
    plot(w1,w2,'or')
    plot(w1,w2,'k','linewidth',2)
    plot(w1',w2','k','linewidth',2)
    hold off
    title(['t=' num2str(t)]);
    pause(0.005)
    end

