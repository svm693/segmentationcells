% Samuel Vilchez
% 2018-04-30
% Semantic Segmentation using Recurrent Convolutional Neural Networks
% example: https://www.mathworks.com/help/vision/ug/semantic-segmentation-examples.html#mw_6ab02754-d2fa-4330-8bea-3eeec77279da
clc, clearvars


% modify dir to main folder containing original images as well as pixelLabelData
dir={''};
imDir=fullfile(dir,'Images');
labelDir=fullfile(dir,'PixelLabelData');
NewimDir=fullfile(dir,'Partitioned','Images');
NewlabelDir=fullfile(dir,'Partitioned','PixelLabelData');

cd(char(NewimDir))
if exist('I1.jpg', 'file') == 2

else
    I=imread(fullfile(char(dir),'RT4-182.tif'));
    I=im2single(I);

    % divide images into 4 smaller images for training and validation
    [a,b,~]=size(I);
    I1=I(1:a/2,1:a/2); %top left
    I2=I((a/2)+1:a,1:a/2); %bottom left
    I3=I(1:a/2,a/2+1:a); %top right
    I4=I(a/2+1:a,a/2+1:a); %bottom right

    imwrite(I1,'I1.jpg');
    imwrite(I2,'I2.jpg');
    imwrite(I3,'I3.jpg');
    imwrite(I4,'I4.jpg');
end



cd(char(NewlabelDir))
if exist('L1.png','file')==2

else
    L=imread(fullfile(char(labelDir),'Label_223.png')); 
    L=im2single(L);
    
    L1=L(1:a/2,1:a/2); %top left
    L2=L((a/2)+1:a,1:a/2); %bottom left
    L3=L(1:a/2,a/2+1:a); %top right
    L4=L(a/2+1:a,a/2+1:a); %bottom right
    
    imwrite(L1,'L1.png');
    imwrite(L2,'L2.png');
    imwrite(L3,'L3.png');
    imwrite(L4,'L4.png');
end

cd(char(dir))
% re-create ground truth for image I and label L
% load ground truth with labeled image 223 (rt4-182)
load('F:\Box Sync\Bioengineering\Research\Dr. Wong Lab\Data\MATLAB\SCM-Sandbox\RCNN-Segm\RCNN-SCM-gTruth\Try-1image\gTruth-1.mat');
labelDefs=gTruth.LabelDefinitions;
classes=labelDefs.Name;
pxds=pixelLabelDatastore(NewlabelDir,classes,labelDefs.PixelLabelID);
imds=imageDatastore(NewimDir);

% confirm and display
Im = readimage(imds,2);
C = readimage(pxds,2);
L=uint8(C);
%imshowpair(Im,L,'montage')

% create and partition data
allData=pixelLabelImageDatastore(imds,pxds);
trainingData=partitionByIndex(allData,[1 3 4]);
valData=partitionByIndex(allData,2);

% To adjust class inbalance
% view: https://www.mathworks.com/examples/matlab-computer-vision/mw/vision-ex41887170-semantic-segmentation-using-deep-learning?s_tid=examples_p1_MLT
tbl = countEachLabel(pxds);

% % visualization
% frequency = tbl.PixelCount/sum(tbl.PixelCount);
% bar(1:numel(classes),frequency)
% xticks(1:numel(classes)) 
% xticklabels(tbl.Name)
% xtickangle(45)
% ylabel('Frequency')

% balance classes with class weighting
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights);


%--------------------- create neural network from matlab example------
%--------------------- method 1---------------------------------------
% link: https://www.mathworks.com/help/vision/ug/semantic-segmentation-examples.html#mw_6ab02754-d2fa-4330-8bea-3eeec77279da

%create Image input layer
% view: https://www.mathworks.com/help/nnet/ref/nnet.cnn.layer.imageinputlayer.html
[h,w,c]=size(Im);
imgLayer=imageInputLayer([h w c]);

% view: https://www.mathworks.com/help/vision/ref/pixellabelimagedatastore.html
numFilters = 64;
filterSize = 3;
numClasses = 3;
BFsegNet = [
    imgLayer
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pxLayer()
    ]
analyzeNetwork(BFsegNet)


% setup training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',100, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', 'GPU', ...
    'ValidationData',valData);

%clear unused variables
clear numFilters filterSize numClasses h w c Im C L tbl imageFreq classWeights pxLayer imgLayer
gpuDevice(1)
tBFseg=trainNetwork(trainingData,BFsegNet,opts);
 


% test net through imds and pxds
testI=readimage(imds,4);
testC=semanticseg(testI,tBFseg);
B=labeloverlay(testI,testC);
figure
imshow(B)


