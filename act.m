function [nextState, reward, done] = act(state, n, m, action)
% action 1 left, 2 right, 3 up, 4 down

M = reshape(state(1:n * m), n, m);
apple = state(n * m + 1:end);
len = length(find(M > 0));
[x y] = find(M == len);

nM = max(M - 1, 0);
done = 0;
reward = -1;
nx = x; ny = y;

if action == 1
    ny -= 1;
elseif action == 2
    ny += 1;
elseif action == 3
    nx -= 1;
elseif action == 4
    nx += 1;
end

% edge or body
if or(nx < 1, nx > n, ny < 1, ny > m)
    disp("Edge Crash !");
    reward = 0;
    done = 1;
    nextState = - ones(n * m + 2, 1);
    return
end

if nM(nx, ny) != 0
    disp("Body Crash !");
    reward = 0;
    done = 1;
    nextState = - ones(n * m + 2, 1);
    return
end

% fprintf("At: (%d, %d)\n", nx, ny);

if and(apple(1) == nx, apple(2) == ny)
    reward = n + len;
    nM = M;
    nM(nx, ny) = len + 1;
    apple = genApple(nM, n);
    disp(size(apple));
    fprintf("You've ate an apple ! (regenerate at %d, %d)\n", apple(1), apple(2));
else
    nM(nx, ny) = len;
    reward = 2 - (abs(nx - apple(1)) + abs(ny - apple(2))) / n;
end

nextState = [nM(:); apple(:)];

end