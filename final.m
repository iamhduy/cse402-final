imageDir = './Celebrity Face Datasets/dataset/';
d = dir(imageDir);
isSubdir = [d.isdir];
subdirs = d(isSubdir);
subdirs = subdirs(~ismember({subdirs.name}, {'.', '..'}));

for k = 1:numel(subdirs)
    folderName = subdirs(k).name;
    disp(folderName);
end


