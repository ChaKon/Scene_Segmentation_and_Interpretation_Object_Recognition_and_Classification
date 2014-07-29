function pascalproject
%this adds the toolbox to the matlab path
%url with the info to make the path permanent: http://www.vlfeat.org/install-matlab.html
run('vlfeat-0.9.16-bin\vlfeat-0.9.16\toolbox\vl_setup.m');
vl_version verbose
% paths that needed
addpath([cd '\VOCcode\']);
rmpath('E:\Education\2nd Semester\PROJECTS\Scene Segmentation and Interpretation\Project\prtools\prtools\');
% addpath('E:\Education\2nd Semester\PROJECTS\Scene Segmentation and Interpretation\Project\prtools\prtools\');
% addpath('E:\Education\2nd Semester\PROJECTS\Scene Segmentation and Interpretation\Project\Matlab_VOCdevkit_2006\VOCdevkit\VOCcode\');
% addpath('E:\Education\2nd Semester\PROJECTS\Scene Segmentation and Interpretation\Project\VOCdevkit\VOC2006\');

VOCinit;

set_imgclass=[5 10 15]; % image classes
set_numclusters=[50 100 200 400]; % Number of clusters
% Train and Test Classifier for each class
for nc=1:size(set_numclusters,2)
    for ic=1:size(set_imgclass,2);
        numclusters=set_numclusters(1,nc);
        imgclass=set_imgclass(1,ic);
        try 
            load(sprintf('./bag_words%dclusters%dimagesclass.mat',numClusters,numImages_class));
        catch
         bag_words=createbow(VOCopts,numclusters,imgclass);   
        end   
%         addpath('E:\Education\2nd Semester\PROJECTS\Scene Segmentation and Interpretation\Project\prtools\prtools\');
        for i=1:VOCopts.nclasses
            cls=VOCopts.classes{i};
            try
                load(sprintf('./matfiles/classifier%s%dclusters%dimagesclass.mat',cls,numclusters,imgclass));
            catch
            disp('creating the training classifier');
            classifier=training(VOCopts,cls,bag_words);% train classifier
            end
            testing(VOCopts,cls,classifier,bag_words);                   % test classifier
            [fp,tp,auc]=VOCroc(VOCopts,'comp1',cls,true);   % compute and display ROC
     rmpath('E:\Education\2nd Semester\PROJECTS\Scene Segmentation and Interpretation\Project\prtools\prtools\');
    if i<VOCopts.nclasses
        fprintf('press any key to continue with next class...\n');
        pause;
    end
        end
    end
end
% Dictionary or bag of words
% load 'train' image set for class
function bag_words=createbow(VOCopts,numclusters,imgclass)
disp('Creating bag of words');
sift_desc=[];
for class=1:VOCopts.nclasses % for all classes
    cls=VOCopts.classes{class};
[idsg,classifierbow.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
% extract features for each image
classifierbow.FD=zeros(0,length(idsg));
tic;
count=0;
num=1;
while(count<=imgclass && num<length(idsg))
    classifierbow.gt(num);
    if (classifierbow.gt(num)==1)
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,idsg{num}),'fd');
    catch
        % compute and save features
        fprintf('%s: train: %d/%d\n',cls,num,length(idsg));
        drawnow;
        bowI=imread(sprintf(VOCopts.imgpath,idsg{num}));
        fd=extractfd(VOCopts,bowI);
        save(sprintf(VOCopts.exfdpath,idsg{num}),'fd');
    end
    % getting bag of words
    sift_desc=[sift_desc;fd'];
    count=count+1;
    end
    num=num+1;
end
end
sift_desc=double(sift_desc);
disp('sift descriptors are obtained in bag of words');
% using k means
disp('Now we are in K-means for Bag_words');
[clusterids clustercenter]=kmeans(sift_desc,numclusters);
disp('k-means in done');
% center of cluster to bag of words
bag_words=clustercenter;
save(sprintf('./matfiles/bag_words%dclusters%dimagesclass.mat',numclusters,imgclass),'bag_words');
% Training the data set
function classifier=training(VOCopts,cls,bag_words)
% Using training image data set
[ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
% extract features for each image
classifier.FD=zeros(length(ids),0);
tic;
for i=1:length(ids)
    % display progress
    disp('you are in training at step in the iteration');
    i
    disp('Now you are training the image');
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end

    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        fd=extractfd(VOCopts,I);
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end
    % Histograms representation
    fd=double(fd);
    histo=histograms(VOCopts,fd',bag_words);
    classifier.FD(i,1:length(histo))=histo;
end
 addpath('E:\Education\2nd Semester\PROJECTS\Scene Segmentation and Interpretation\Project\prtools\prtools\');
 data=dataset(classifier.FD,classifier.gt);
 % svm has to apply
 classifier=svc(data); % SVM classifier
%  classifier=adaboostc(data); %adaboost classfier


% getting feature using sift
function fd=extractfd(VOCopts,I)
disp('you are extracting the features');
Igray=rgb2gray(I);
s_Igray=single(Igray);
[f d]=vl_sift(s_Igray);
% Ref http://www.vlfeat.org/overview/sift.html
% peak threshold filters peaks of the DoG scale space that are too small (in absolute value).
% edge threshold eliminates peaks of the DoG scale space whose curvature is
% too small.
% to increase the fetures we can increase either PeakThresh and edgethresh
fd=d;

% 
function histo=histograms(VOCopts,descrip,bag_words)
% Using k nearest neighborhood for classification
disp('doing histogram');
a=length(descrip);
b=length(bag_words);
class=knnclassify(descrip,bag_words, 1:size(bag_words,1),50);
histo=zeros(1,size(bag_words));
for h=1:size(class,1)
    histo(1,class(h))=histo(1,class(h))+1;
end
% normalizing
histo=histo/sum(histo);
% run classifier on test images
function testing(VOCopts,cls,classifier,bag_words)
disp('you are in testing step');
% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');

% create results file
fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');

% classify each image
tic;
for i=1:length(ids)
    % display progress
    disp('you are in testing step in iteration');
    i
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        fd=extractfd(VOCopts,I);
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end
    fd=double(fd);
    histo=histograms(VOCopts,fd',bag_words);
    % compute confidence of positive classification
    c=classify(VOCopts,classifier,histo);
    
    % write to results file
    fprintf(fid,'%s %f\n',ids{i},c);
end

% close results file
fclose(fid);

% trivial classifier: compute ratio of L2 distance betweeen
% nearest positive (class) feature vector and nearest negative (non-class)
% feature vector
function c = classify(VOCopts,classifier,histo)

% d=sum(fd.*fd)+sum(classifier.FD.*classifier.FD)-2*fd'*classifier.FD;
% dp=min(d(classifier.gt>0));
% dn=min(d(classifier.gt<0));
% c=dn/(dp+eps
cstr = struct(histo*classifier*classc);
c = cstr.data(end);

    

