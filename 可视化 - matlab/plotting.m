%% cutmix
figure;
subplot(2,1,1);
drawlosscurve(CutMix_LeNet);
title('Loss Curve - CutMix LeNet');

subplot(2,1,2);
drawacccurve(CutMix_LeNet);
title('Acc Curve - CutMix LeNet');

figure;
subplot(2,1,1);
drawlosscurve(CutMix_ResNet);
title('Loss Curve - CutMix ResNet');

subplot(2,1,2);
drawacccurve(CutMix_ResNet);
title('Acc Curve - CutMix ResNet');

figure;
subplot(2,1,1);
drawlosscurve(CutMix_VGG);
title('Loss Curve - CutMix VGG');

subplot(2,1,2);
drawacccurve(CutMix_VGG);
title('Acc Curve - CutMix VGG');

%% mixup
figure;
subplot(2,1,1);
drawlosscurve(MixUp_LeNet);
title('Loss Curve - MixUp LeNet');

subplot(2,1,2);
drawacccurve(MixUp_LeNet);
title('Acc Curve - MixUp LeNet');

figure;
subplot(2,1,1);
drawlosscurve(MixUp_ResNet);
title('Loss Curve - MixUp ResNet');

subplot(2,1,2);
drawacccurve(MixUp_ResNet);
title('Acc Curve - MixUp ResNet');

figure;
subplot(2,1,1);
drawlosscurve(MixUp_VGG);
title('Loss Curve - MixUp VGG');

subplot(2,1,2);
drawacccurve(MixUp_VGG);
title('Acc Curve - MixUp VGG');

%% cutout
figure;
subplot(2,1,1);
drawlosscurve(CutOut_LeNet);
title('Loss Curve - CutOut LeNet');

subplot(2,1,2);
drawacccurve(CutMix_LeNet);
title('Acc Curve - CutOut LeNet');

figure;
subplot(2,1,1);
drawlosscurve(CutOut_ResNet);
title('Loss Curve - CutOut ResNet');

subplot(2,1,2);
drawacccurve(CutOut_ResNet);
title('Acc Curve - CutOut ResNet');

figure;
subplot(2,1,1);
drawlosscurve(CutOut_VGG);
title('Loss Curve - CutOut VGG');

subplot(2,1,2);
drawacccurve(CutOut_VGG);
title('Acc Curve - CutOut VGG');

%% baseline
figure;
subplot(2,1,1);
drawlosscurve(BaseLine_LeNet);
title('Loss Curve - BaseLine LeNet');

subplot(2,1,2);
drawacccurve(BaseLine_LeNet);
title('Acc Curve - BaseLine LeNet');

figure;
subplot(2,1,1);
drawlosscurve(BaseLine_ResNet);
title('Loss Curve - BaseLine ResNet');

subplot(2,1,2);
drawacccurve(BaseLine_ResNet);
title('Acc Curve - BaseLine ResNet');

figure;
subplot(2,1,1);
drawlosscurve(MixUp_VGG);
title('Loss Curve - BaseLine VGG');

subplot(2,1,2);
drawacccurve(BaseLine_VGG);
title('Acc Curve - BaseLine VGG');

%% compare acc - based on same baseline
figure;
epoch = 200;
plot([1:epoch],BaseLine_LeNet.testAcc,'LineWidth',2);
hold on;
plot([1:epoch],CutOut_LeNet.testAcc,'LineWidth',2);
plot([1:epoch],MixUp_LeNet.testAcc,'LineWidth',2);
plot([1:epoch],CutMix_LeNet.testAcc,'LineWidth',2);
xlabel('epoch');
ylabel('test accuracy(%)');
legend('No Augmentation','CutOut','MixUp','CutMix');
grid on;
title('��ͬ������ǿ��ʽ��LeNet-5�ϵı���');

figure;
plot([1:75],BaseLine_VGG.testAcc,'LineWidth',2);
hold on;
plot([1:89],CutOut_VGG.testAcc,'LineWidth',2);
plot([1:100],MixUp_VGG.testAcc,'LineWidth',2);
plot([1:77],CutMix_VGG.testAcc,'LineWidth',2);
xlabel('epoch');
ylabel('test accuracy(%)');
legend('No Augmentation','CutOut','MixUp','CutMix');
grid on;
title('��ͬ������ǿ��ʽ��VGG19�ϵı���');

figure;
plot([1:10],BaseLine_ResNet.testAcc,'LineWidth',2);
hold on;
plot([1:10],CutOut_ResNet.testAcc,'LineWidth',2);
plot([1:20],MixUp_ResNet.testAcc,'LineWidth',2);
plot([1:20],CutMix_ResNet.testAcc,'LineWidth',2);
xlabel('epoch');
ylabel('test accuracy(%)');
legend('No Augmentation','CutOut','MixUp','CutMix');
grid on;
title('��ͬ������ǿ��ʽ��ResNet�ϵı���');

%% compare acc - based on same augmentation
figure;
plot([1:size(BaseLine_LeNet,1)],BaseLine_LeNet.testAcc,'LineWidth',2);
hold on;
plot([1:size(BaseLine_VGG,1)],BaseLine_VGG.testAcc,'LineWidth',2);
plot([1:size(BaseLine_ResNet,1)],BaseLine_ResNet.testAcc,'LineWidth',2);
xlabel('epoch');
ylabel('test accuracy(%)');
legend('LeNet-5','VGG19','ResNet');
grid on;
title('����������ǿʱ����Baseline�����ܱȽ�');

figure;
plot([1:size(CutOut_LeNet,1)],CutOut_LeNet.testAcc,'LineWidth',2);
hold on;
plot([1:size(CutOut_VGG,1)],CutOut_VGG.testAcc,'LineWidth',2);
plot([1:size(CutOut_ResNet,1)],CutOut_ResNet.testAcc,'LineWidth',2);
xlabel('epoch');
ylabel('test accuracy(%)');
legend('LeNet-5','VGG19','ResNet');
grid on;
title('ʹ��CutOut����ʱ����Baseline�����ܱȽ�');

figure;
plot([1:size(MixUp_LeNet,1)],MixUp_LeNet.testAcc,'LineWidth',2);
hold on;
plot([1:size(MixUp_VGG,1)],MixUp_VGG.testAcc,'LineWidth',2);
plot([1:size(MixUp_ResNet,1)],MixUp_ResNet.testAcc,'LineWidth',2);
xlabel('epoch');
ylabel('test accuracy(%)');
legend('LeNet-5','VGG19','ResNet');
grid on;
title('ʹ��MixUp����ʱ����Baseline�����ܱȽ�');

figure;
plot([1:size(CutMix_LeNet,1)],CutMix_LeNet.testAcc,'LineWidth',2);
hold on;
plot([1:size(CutMix_VGG,1)],CutMix_VGG.testAcc,'LineWidth',2);
plot([1:size(CutMix_ResNet,1)],CutMix_ResNet.testAcc,'LineWidth',2);
xlabel('epoch');
ylabel('test accuracy(%)');
legend('LeNet-5','VGG19','ResNet');
grid on;
title('ʹ��CutMix����ʱ����Baseline�����ܱȽ�');

% %% hotmap
% figure;
% xvalues = {'No Aug','CutOut','MixUp','CutMix'};
% yvalues = {'LeNet-5','VGG19','ResNet'};
% mtrix = [max(BaseLine_LeNet.testAcc),max(CutOut_LeNet.testAcc),...,
%     max(MixUp_LeNet.testAcc),max(CutMix_LeNet.testAcc);...,
%     max(BaseLine_VGG.testAcc),max(CutOut_VGG.testAcc),...,
%     max(MixUp_VGG.testAcc),max(CutMix_VGG.testAcc);...,
%     max(BaseLine_ResNet.testAcc),max(CutOut_ResNet.testAcc),...,
%     max(MixUp_ResNet.testAcc),max(CutMix_ResNet.testAcc)];
% heatmap(xvalues,yvalues,mtrix);
% xlabel('best accuracy(%)');


%% loss curve function
function curve = drawlosscurve(aTable)
epoch = size(aTable,1);
plot([1:epoch],aTable.trainLoss,'LineWidth',2);
hold on;
plot([1:epoch],aTable.testLoss,'LineWidth',2);
xlabel('epoch');
ylabel('loss');
legend('train','test');
grid on;
end

%% acc curve function
function curve = drawacccurve(aTable)
epoch = size(aTable,1);
plot([1:epoch],aTable.trainAcc,'LineWidth',2);
hold on;
plot([1:epoch],aTable.testAcc,'LineWidth',2);
xlabel('epoch');
ylabel('acc(%)');
legend('train','test');
grid on;
end
