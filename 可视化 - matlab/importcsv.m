CutMix_LeNet = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/CutMix_LeNet.csv");
CutMix_ResNet1 = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/CutMix_ResNet.csv");
CutMix_VGG = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/CutMix_VGG.csv");
MixUp_LeNet = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/MixUp_LeNet.csv");
MixUp_ResNet = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/MixUp_ResNet.csv");
MixUp_VGG = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/MixUp_VGG.csv");
CutOut_LeNet = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/CutOut_LeNet.csv");
CutOut_ResNet = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/CutOut_ResNet.csv");
CutOut_VGG = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/CutOut_VGG.csv");
BaseLine_LeNet = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/BaseLine_LeNet.xlsx");
BaseLine_ResNet = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/BaseLine_ResNet.csv");
BaseLine_VGG = readtable("/Users/yrzhu/Desktop/课程/大三/大三下/计算机视觉/期末pj/plotting/BaseLine_VGG.csv");

CutMix_ResNet = table([1:20]',zeros(20,1),zeros(20,1),zeros(20,1),zeros(20,1));
CutMix_ResNet.Properties.VariableNames = {'epoch','trainAcc','testAcc','trainLoss','testLoss'};
a = size(CutMix_ResNet1.trainAcc,1)/20;
b = 6260/20;
for i = 0:19
    CutMix_ResNet.trainAcc(i+1) = mean(CutMix_ResNet1.trainAcc([(a*i+1):(a*(i+1))]));
    CutMix_ResNet.testAcc(i+1) = mean(CutMix_ResNet1.testAcc([(b*i+1):(b*(i+1))]));
    CutMix_ResNet.trainLoss(i+1) = mean(CutMix_ResNet1.trainLoss([(a*i+1):(a*(i+1))]));
    CutMix_ResNet.testLoss(i+1) = mean(CutMix_ResNet1.testLoss([(b*i+1):(b*(i+1))]));
end

%%
CutOut_LeNet.trainAcc = CutOut_LeNet.trainAcc*100;
CutOut_ResNet.trainAcc = CutOut_ResNet.trainAcc*100;
CutOut_VGG.trainAcc = CutOut_VGG.trainAcc*100;
BaseLine_LeNet.trainAcc = BaseLine_LeNet.trainAcc*100;
BaseLine_ResNet.trainAcc = BaseLine_ResNet.trainAcc*100;
BaseLine_VGG.trainAcc = BaseLine_VGG.trainAcc*100;

CutOut_LeNet.testAcc = CutOut_LeNet.testAcc*100;
CutOut_ResNet.testAcc = CutOut_ResNet.testAcc*100;
CutOut_VGG.testAcc = CutOut_VGG.testAcc*100;
BaseLine_LeNet.testAcc = BaseLine_LeNet.testAcc*100;
BaseLine_ResNet.testAcc = BaseLine_ResNet.testAcc*100;
BaseLine_VGG.testAcc = BaseLine_VGG.testAcc*100;