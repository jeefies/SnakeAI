function [J grad] = costFunction(W, layers, X, Y)

% disp("Calc Cost...");
% disp(layers);

m = size(X, 1);
% 103 * 25 + 26 * 10 + 11 * 4 = 2879
% 103 * 25 + 26 * 4 = 2679
if length(layers) == 4
    W1 = reshape(W(1:layers(2) * (layers(1) + 1)), layers(2), layers(1) + 1);
    W2 = reshape(W(1:layers(3) * (layers(2) + 1)), layers(3), layers(2) + 1);
    W3 = reshape(W(1:layers(4) * (layers(3) + 1)), layers(4), layers(3) + 1);

    % forward propagation

    A1 = max([ones(m, 1) X] * W1', 0);
    A2 = max([ones(m, 1) A1] * W2', 0);
    A3 = [ones(m, 1) A2] * W3';

    % disp([A3 Y]);

    J = sum(sum((A3 - Y) .^ 2)) / m;
    % fprintf("J %f\n", J);

    % backward propagation

    D1 = zeros(size(W1)); % 25 x 103
    D2 = zeros(size(W2));
    D3 = zeros(size(W3));

    for i = 1:m
        a0 = X(i, :);  % 1 x 102
        a1 = A1(i, :); % 1 x 25
        a2 = A2(i, :); % 1 x 10
        a3 = A3(i, :); % 1 x 4
        y = Y(i, :);   % 1 x 4

        d3 = (a3 - y)';   % 4 x 1
        d2 = (W3' * d3) .* ([1 a2]' > 0);  % 11 x 1
        d2 = d2(2:end); % 10 x 1
        d1 = (W2' * d2) .* ([1 a1]' > 0); % 26 x 1
        d1 = d1(2:end); % 25 x  1

        D1 += d1 * [1 a0];
        D2 += d2 * [1 a1];
        D3 += d3 * [1 a2];
    end

    grad = [D1(:); D2(:); D3(:)] / m;
end

% disp(grad');

% function M = softPlus(X)
%     M = log(1 + exp(X));
% end

% function M = softPlusGradient(X)
%     M = 1 ./ (1 + exp(-X));
% end

% 28 * 10 + 11 * 4 = 280 + 44 = 324
if length(layers) == 3
    W1 = reshape(W(1:layers(2) * (layers(1) + 1)), layers(2), layers(1) + 1);
    W2 = reshape(W(1:layers(3) * (layers(2) + 1)), layers(3), layers(2) + 1);

    % forward propagation
    A1 = ReLU([ones(m, 1) X] * W1');
    % A1
    A2 = [ones(m, 1) A1] * W2';
    % A2
    J = sum(sum((A2 - Y) .^ 2)) / (2 * m);

    % backward propagation
    D1 = zeros(size(W1)); % 25 x 103
    D2 = zeros(size(W2));

    for i = 1:m
        a0 = X(i, :);  % 1 x 27
        a1 = A1(i, :); % 1 x 10
        a2 = A2(i, :); % 1 x 4
        y = Y(i, :);   % 1 x 4

        d2 = (a2 - y)' .* ReLUGradient(a2'); % 4 x 1
        d1 = (W2' * d2) .* ReLUGradient([1 a1]'); % 26 x 1
        d1 = d1(2:end); % 25 x 1

        % d1
        % d2

        D1 += d1 * [1 a0];
        D2 += d2 * [1 a1];
    end

    % disp(D2(:)');

    grad = [D1(:); D2(:)] / m;
end

end

% A1 =

%  Columns 1 through 7:

%    8.2010e-03   7.8839e-03   1.1086e-02   8.4722e-03   1.3617e-02   6.1709e-03   1.2482e-02

%  Columns 8 through 10:

%    6.4490e-03   1.4662e-02   6.5967e-03

% A2 =

%    1.7542e-03   4.7400e-04   1.6264e-03   7.8085e-04

% d1 =

%    5.4902e-07
%    4.5324e-07
%    4.0452e-07
%    3.1031e-07
%    1.8260e-07
%    4.4165e-07
%    5.3310e-07
%    2.4923e-07
%    6.1534e-07
%    3.6067e-07

% d2 =

%             0
%             0
%    3.3528e-04
%             0