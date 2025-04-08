%% Parameters
% Directory with your results
%%% Make sure the file names are as exactly %%%
%%% as the original ground truth images %%%
dir_path_group = {
    ['../SCI-main_cvpr2022light/unpair_data/MEF/'],
%     '../haze/',
%  '../MSNet/',
%     '../EPDN/',
%     '../DCPDN/',
%     '../GFN/',
%     '../NLD/',
%     '../PFFN/',
%     '../DCP/',
%     '../DehazeNet/',
%     '../dehaze-cGAN/',
}
for i = 1:1:4
    dir_path = dir_path_group{i}
    input_dir = fullfile(pwd,dir_path);

    % Directory with ground truth images
    %GT_dir = fullfile(pwd,'TTTTT/TTTTT/self_validation/Urban100/');

    % Number of pixels to shave off image borders when calcualting scores
    shave_width = 16;

    % Set verbose option
    verbose = true;

    %% Calculate scores and save
    addpath utils
    scores = calc_scores(input_dir,input_dir,shave_width,verbose);

    % Saving
    save([dir_path,'your_scores.mat'],'scores')

    %% Printing results
    perceptual_score = (mean([scores.NIQE]) + (10 - mean([scores.Ma]))) / 2;
    fprintf(['\n\nYour NIQE is: ',num2str(mean([scores.NIQE]))]);
    fprintf(['\n\nYour Ma is: ',num2str(mean([scores.Ma]))]);
    fprintf(['\n\nYour perceptual score is: ',num2str(perceptual_score)]);
    fprintf(['\nYour RMSE is: ',num2str(sqrt(mean([scores.MSE]))),'\n']);
end