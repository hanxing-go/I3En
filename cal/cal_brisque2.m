model = brisqueModel(rand(10,1), 10, rand(10,36), 0.25);
dir_path_group = {
      '../ESRGAN/',
%       '../haze/',
%       '../MSNet/',
%       '../EPDN/',
%       '../DCPDN/',
%       '../GFN/',
%       '../NLD/',
%       '../PFFN/',
%       '../DCP/',
%       '../DehazeNet/',
%       '../dehaze-cGAN/',
%       '../AODNet/'
}
for i = 1:1:11
    dir_path = dir_path_group{i}
    input_dir = fullfile(pwd,dir_path);
    input_file_list = dir([input_dir,'/*.png']);
    im_num = length(input_file_list)
    scores = struct([]);
    %yimy = 0;
    for ii=1:im_num
        input_file_list(ii).name
        input_image_path = fullfile(input_dir,input_file_list(ii).name);
        scores(ii).brique = brisque(imread(input_image_path), model);
        scores(ii).name = input_file_list(ii).name;
        %yimy += scores(ii).brique;
    end
    %fprintf(['\n\nYour brisque is: ',num2str(yimy/im_num)]);
    fprintf(['\n\nYour brisque is: ',num2str(mean([scores.brique]))]);
    save([dir_path,'brisque_scores.mat'],'scores')

end