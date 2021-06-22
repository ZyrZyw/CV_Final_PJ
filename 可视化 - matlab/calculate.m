tablelist = {BaseLine_LeNet.testAcc,BaseLine_VGG.testAcc,BaseLine_ResNet.testAcc,...,
    CutOut_LeNet.testAcc,CutOut_VGG.testAcc,CutOut_ResNet.testAcc,...,
    MixUp_LeNet.testAcc,MixUp_VGG.testAcc,MixUp_ResNet.testAcc,...,
    CutMix_LeNet.testAcc,CutMix_VGG.testAcc,CutMix_ResNet.testAcc};

avgtop5acc = [];
for i = 1:12
    tab = tablelist{i};
    a = sort(tab,'descend');
    avgtop5acc(i) = mean(a(1:5));
end

releperformance = [];
absperforamnce = [];
for k = 1:3
    releperformance(1,k) = 0;
    releperformance(2,k) = (avgtop5acc(k+3)-avgtop5acc(k))/avgtop5acc(k);
    releperformance(3,k) = (avgtop5acc(k+6)-avgtop5acc(k))/avgtop5acc(k);
    releperformance(4,k) = (avgtop5acc(k+9)-avgtop5acc(k))/avgtop5acc(k);
    for j = 1:4
        absperforamnce(j,k) = avgtop5acc(3*j-3+k);
    end
end
releperformance = releperformance';

%% hotmap - abs
xvalues = {'No Aug','CutOut','MixUp','CutMix'};
yvalues = {'LeNet-5','VGG19','ResNet'};
% mtrix = [max(BaseLine_LeNet.testAcc),max(CutOut_LeNet.testAcc),...,
%     max(MixUp_LeNet.testAcc),max(CutMix_LeNet.testAcc);...,
%     max(BaseLine_VGG.testAcc),max(CutOut_VGG.testAcc),...,
%     max(MixUp_VGG.testAcc),max(CutMix_VGG.testAcc);...,
%     max(BaseLine_ResNet.testAcc),max(CutOut_ResNet.testAcc),...,
%     max(MixUp_ResNet.testAcc),max(CutMix_ResNet.testAcc)];
heatmap(xvalues,yvalues,absperforamnce');
xlabel('top5 accuracy(ave)(%)');

