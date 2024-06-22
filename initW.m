%% INIT W
epsilon = 0.001;
Qnet = rand(324, 1) * epsilon * 2;
save("Qnet.mat", "Qnet");

% 2879

% layers = [27, 16, 8, 4];
% 28 * 16 + 17 * 8 + 9 * 4 = 448 + 136 + 36 = 620