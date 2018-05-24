% Samuel Vilchez
% 2018-04-30
% Semantic Segmentation using Recurrent Convolutional Neural Networks

% License
%{
%   =======================================================================================
%   Copyright (C) 2018  Samuel Vilchez
%   Email: samvilchez@gmail.com
%
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
% 
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
% 
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%   =======================================================================================
%}


clc, clearvars

% Create ground truth
dir={'F:\Box Sync\Bioengineering\Research\Dr. Wong Lab\Data\MATLAB\segmentationcells-Sandbox'};
imDir=fullfile(dir,'Images');
labelDir=fullfile(dir,'PixelLabelData');

cd(char(imDir))
if exist('I111.png', 'file') == 2

else
    cd(char(dir))
    I{1}=imread('RT4-180.tif');
    I{2}=imread('RT4-182.tif');
    
    [p,q,~]=size(I)

    for k=1:q
          
        I{k}=im2single(I{k});  

        % divide images into non-overlapping blocks of 568 by 568
        n=568;
        [a,b,~]=size(I{k});
        B=cell(a/n,b/n);
        cd(char(imDir))
        
        i=1;
        ci=0;
        for i = 1:n:a-(n-1)
            ci = ci + 1;
            j=1;
            cj=0;
            for j = 1:n:b-(n-1)
                cj = cj + 1;
                B{ci,cj} = I{k}(i:i+(n-1),j:j+(n-1));
                name=sprintf('I%d%d%d.png',k,ci,cj);
                imwrite(B{ci,cj},name)
            end
        end 
    end
end

cd(char(labelDir))
if exist('L111.png','file')==2

else
    cd(char(dir))
    L{1}=imread('Label_180.png');
    L{2}=imread('Label_182.png');
    
    [p,q,~]=size(L)

    for k=1:q       
        
        L{k}=im2single(L{k});  

        % divide images into non-overlapping blocks of 568 by 568
        n=568;
        [a,b,~]=size(L{k});
        B=cell(a/n,b/n);
        cd(char(labelDir))
        
        i=1;
        ci=0;
        for i = 1:n:a-(n-1)
            ci = ci + 1;
            j=1;
            cj=0;
            for j = 1:n:b-(n-1)
                cj = cj + 1;
                B{ci,cj} = L{k}(i:i+(n-1),j:j+(n-1));
                name=sprintf('L%d%d%d.png',k,ci,cj);
                imwrite(B{ci,cj},name)
            end
        end 
    end
end


%%

cd(char(dir))
% convert unit8 to single for MATLAB processing

% re-create ground truth for image I and label L
load('F:\Box Sync\Bioengineering\Research\Dr. Wong Lab\Data\MATLAB\SCM-Sandbox\RCNN-Segm\RCNN-SCM-gTruth\Trial\gTruth2im.mat');
labelDefs=gTruth.LabelDefinitions;
classes=labelDefs.Name;
pxds=pixelLabelDatastore(labelDir,classes,labelDefs.PixelLabelID);
imds=imageDatastore(imDir);

%Confirm and display
Im = readimage(imds,2);
C = readimage(pxds,2);
C2=uint8(C);
imshowpair(Im,C2,'montage')

% create and partition data
allData=pixelLabelImageDatastore(imds,pxds);
% partition data in 75/25... 96 images for training and 128-96=32 for
% validation
%---------------------
tVector=randperm(128,96);
vVector=zeros(1,32);
j=1;
for i=1:128
    num=find(tVector == i);
   if num > 0
   
   else
      vVector(j)=i;
      j=j+1;
   end    
end

trainingData=partitionByIndex(allData,tVector);
valData=partitionByIndex(allData,vVector);

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
    ];
analyzeNetwork(BFsegNet)


% setup training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',100, ...
    'MiniBatchSize',2, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', 'GPU', ...
    'ValidationData',valData);

%clear unused variables
clear numFilters filterSize numClasses h w c Im C L tbl imageFreq classWeights pxLayer imgLayer
gpuDevice(1)
tBFseg=trainNetwork(trainingData,BFsegNet,opts);
 


% test net through imds and pxds
% view: https://www.mathworks.com/help/vision/ug/semantic-segmentation-examples.html#mw_9ca2a7be-c8c2-4bbf-b168-128261d1be7d

testI=readimage(imds,7);
testC=semanticseg(testI,tBFseg);
B=labeloverlay(testI,testC);
figure
imshow(B)