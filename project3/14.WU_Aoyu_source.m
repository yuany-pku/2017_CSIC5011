%% Load Data
clear
imagefiles = dir('data');  
imagefiles=imagefiles(~ismember({imagefiles.name},{'.','..','.DS_Store'}));
nfiles = length(imagefiles);

% Data Label: 1 True 0 False -1 Disputed
Label = [-1 1 1 1 1 1 -1 1 1 -1 0 0 0 0 0 0 0 0 0 -1 1 1 -1 1 -1 -1 1 1];

% Feature
ImageIntensity = [];
TextureFeature = [];

%% Calculate Feature
for i=1:nfiles
   % Skip Dir
   if imagefiles(i).isdir 
       continue;
   end
   
   % Pre-processing
   img = imread(fullfile(imagefiles(i).folder,imagefiles(i).name));
   I2 = im2double(rgb2gray(img));
   I2F = imgaussfilt(I2);
   
   %% Image Intensity Statistics
   globalmean = mean(I2F(:));
   variance = var(I2F(:));
   hist = histcounts(I2F(:),16);
   skewness_ = skewness(hist);
   kurtosis_ = kurtosis(hist);
   entropy_ = entropy(I2F);
   
   IIS = [globalmean,variance,skewness_,kurtosis_,entropy_];
   ImageIntensity = [ImageIntensity;IIS];
   
   %% Texture Based
   % Gray-level Co-occurrence Matrix 
   glcms = graycomatrix(I2);
   stats = graycoprops(glcms);
   glcmstats = [stats.Contrast,stats.Correlation,stats.Energy,stats.Homogeneity];

   % 2d Discrete Wavelet Transformation
%    [cA,cH,cV,cD] = dwt2(I2,'haar');
   
   TextureFeature = [TextureFeature;glcmstats];
   
end

trainingIdx = find(Label~=-1);
trainingLabel = Label(trainingIdx);

%% Training

% Leave-one-out cross-validation
DTIISuccessCount = 0;
SVMIISuccessCount = 0;
DAIISuccessCount = 0;
kNNIISuccessCount = 0;
DTTBSuccessCount = 0;
SVMTBSuccessCount = 0;
DATBSuccessCount = 0;
kNNTBSuccessCount = 0;
for i = trainingIdx
    CVIdx = trainingIdx(trainingIdx~=i);
    CVLabel = Label(CVIdx);
    
    %% ImageIntensity
    CVIIFeature = ImageIntensity(CVIdx,:);
    % Decision Tree
    DTMdl = fitctree(CVIIFeature,CVLabel);
    if predict(DTMdl,ImageIntensity(i,:))==Label(i)
        DTIISuccessCount = DTIISuccessCount + 1;
    end
    
    % SVM
    SVMMdl = fitcsvm(CVIIFeature,CVLabel);
    if predict(SVMMdl,ImageIntensity(i,:))==Label(i)
        SVMIISuccessCount = SVMIISuccessCount + 1;
    end
        
    % Discriminant Analysis
    DAMdl = fitcsvm(CVIIFeature,CVLabel);
    if predict(DAMdl,ImageIntensity(i,:))==Label(i)
        DAIISuccessCount = DAIISuccessCount + 1;
    end
    
    % kNN
    knnModel = fitcknn(CVIIFeature,CVLabel);
    if predict(knnModel,ImageIntensity(i,:))==Label(i)
        kNNIISuccessCount = kNNIISuccessCount + 1;
    end
    
    %% Texture
    CVTFFeature = TextureFeature(CVIdx,:);
    % Decision Tree
    DTMdl = fitctree(CVTFFeature,CVLabel);
    if predict(DTMdl,TextureFeature(i,:))==Label(i)
        DTTBSuccessCount = DTTBSuccessCount + 1;
    end
    
    % SVM
    SVMMdl = fitcsvm(CVTFFeature,CVLabel);
    if predict(SVMMdl,TextureFeature(i,:))==Label(i)
        SVMTBSuccessCount = SVMTBSuccessCount + 1;
    end
        
    % Discriminant Analysis
    DAMdl = fitcsvm(CVTFFeature,CVLabel);
    if predict(DAMdl,TextureFeature(i,:))==Label(i)
        DATBSuccessCount = DATBSuccessCount + 1;
    end
    
    % kNN
    knnModel = fitcknn(CVTFFeature,CVLabel);
    if predict(knnModel,TextureFeature(i,:))==Label(i)
        kNNTBSuccessCount = kNNTBSuccessCount + 1;
    end
end

%% Test
trainingIdx = find(Label~=-1);
trainingLabel = Label(trainingIdx);
feature = TextureFeature(trainingIdx,:);

m1 = fitctree(feature,trainingLabel);
  m2 = fitcsvm(feature,trainingLabel);
 m3 = fitcknn(feature,trainingLabel);

 trainingIdx = find(Label==-1);
for i = trainingIdx
    i
    predict(m1,TextureFeature(i,:))
    predict(m2,TextureFeature(i,:))
    predict(m3,TextureFeature(i,:))
end
