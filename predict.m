function M = predict(W, layers, X)
m = size(X, 1);

if length(layers) == 4
    W1 = reshape(W(1:layers(2) * (layers(1) + 1)), layers(2), layers(1) + 1);
    W2 = reshape(W(1:layers(3) * (layers(2) + 1)), layers(3), layers(2) + 1);
    W3 = reshape(W(1:layers(4) * (layers(3) + 1)), layers(4), layers(3) + 1);

    A1 = max([ones(m, 1) X] * W1', 0);
    A2 = max([ones(m, 1) A1] * W2', 0);
    M = [ones(m, 1) A2] * W3';
else

    W1 = reshape(W(1:layers(2) * (layers(1) + 1)), layers(2), layers(1) + 1);
    W2 = reshape(W(1:layers(3) * (layers(2) + 1)), layers(3), layers(2) + 1);

    A1 = max([ones(m, 1) X] * W1', 0);
    M = [ones(m, 1) A1] * W2';
end

end