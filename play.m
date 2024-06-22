%% Initialization
clear ; close all; clc

function printMap(state, n, m)
    % disp("Map:");
    M = reshape(state(1:n * m), n, m);
    apple = state(n * m + 1:n * m + 2);
    M(apple(1), apple(2)) = -1;
    % disp(M);
    imagesc(M);
end

% size of the board
% n = 10;
% layers = [102, 25, 10, 4];
n = 5;
layers = [27, 10, 4];
% layers = [102, 25, 4];

function M = flatten(state, nextstate, action, reward, done)
    M = [state(:); nextstate(:); action; reward; done];
end

function [state, nextstate, action, reward, done] = restore(mem, n)
    state = mem(1: n * n + 2);
    nextstate = mem(n * n + 3 : 2 * n * n + 4);
    action = mem(end - 2);
    reward = mem(end - 1);
    done = mem(end);
end

function cluster = sample(mem, cnt)
    m = size(mem, 2);
    idx = randi(m, 1, cnt);
    cluster = mem(:, idx);
    % fprintf("Sample IDX:"); disp(idx);
end

function [Qnet, train] = singleTrain(memory, n, layers, Qnet, Tnet, MIN_REPLAY_SIZE = 1000)
    learnRate = 0.7;
    discountFactor = 0.7;
    train = 0;

    if size(memory, 2) < MIN_REPLAY_SIZE
        % fprintf("Now memory only %d batches !\n", size(memory, 2));
        return
    end


    train = 1;
    fprintf("Single Train !\n");

    batchSize = 1;
    batch = sample(memory, batchSize);

    curStates = [];
    nextStates = [];
    rewards = [];
    dones = [];
    acts = [];

    for i = 1:batchSize
        expe = batch(:, i);
        [cs, ns, ac, rw, dn] = restore(expe, n);
        curStates = [curStates; cs'];
        nextStates = [nextStates; ns'];
        acts = [acts ac];
        rewards = [rewards rw];
        dones = [dones dn];
    end

    curQs = predict(Qnet, layers, curStates);
    savQs = curQs;
    nextQs = predict(Tnet, layers, nextStates);

    for i = 1:batchSize
        done = dones(i);
        reward = rewards(i);
        ac = acts(i);
        if done
            maxQ = reward + discountFactor * max(nextQs(i, :));
        else
            maxQ = reward;
        end

        curQs(i, ac) = (1 - learnRate) * curQs(i, ac) + learnRate * maxQ;
    end

    % disp(size(curStates));

    costF = @(t) costFunction(t, layers, curStates, curQs);
    [cost grad] = costF(Qnet);
    % [_ numgrad] = checkGradients(costF, Qnet);
    % % disp([size(grad) size(numgrad)]);

    % diff = norm(numgrad - grad) / norm(numgrad + grad);
    % % disp([numgrad grad]);
    % printf("Diff Now %f\n", diff);
    % pause;

    fprintf("Old Cost is %f\n", cost);

    MaxIter = 100;

    % options = optimset('MaxIter', MaxIter);
    % [Qnet, cost] = fmincg(costF, Qnet, options);
    % [Qnet, cost] = fminunc(costF, Qnet, options);

    alpha = 0.01;
    for t = 1:30
        [cost grad] = costF(Qnet);
        Qnet -= alpha * grad;
    end

    fprintf("Single Train cost to %f\n", cost);

    % preQs = predict(Qnet, layers, curStates);
    % __eps = 100;
    % disp([round(savQs * __eps) round(curQs * __eps) round(preQs * __eps)]);
end

trainTimes = 10000;

load("Qnet.mat");
Tnet = Qnet;
flushTime = 0;

disp(size(Qnet));


Eps = 1;
maxEps = 1;
minEps = 0.1;
steps = 0;
decay = 1e-2;
timeDelay = 0.02;

memory = [];
verbose = 0;


for t = 1:trainTimes
    if steps > 1000
        % pause;
    end
    if verbose
        pause(1 * timeDelay);
    end

    totalReward = 0;
    trainingRewards = 0;
    
    mmap = zeros(n, n);

    snake = randi(n, 1, 2);
    mmap(snake(1), snake(2)) = 1;
    apple = genApple(mmap, n);

    state = [mmap(:); apple(:)];

    done = 0;
    startStep = steps;
    while not(done)
        steps += 1;

        if verbose
            printMap(state, n, n);
            pause(0.1 * timeDelay);
        end

        randAct = rand(1);
        if randAct <= Eps
            action = randi(4);
            if verbose
                fprintf("Random action %d\n", action);
            end
        else
            Qs = predict(Qnet, layers, state');
            [_, action] = max(Qs);
            if or(verbose, 0)
                fprintf("Q predict action %d\nQ Table:", action);
                disp(Qs);
            end
        end

        [nextstate, reward, done] = act(state, n, n, action);
        % disp(size(nextstate)); disp(size(reward)); disp(size(done)); disp(size(action));
        memory = [flatten(state, nextstate, action, reward, done) memory];
        % disp(size(memory));

        if size(memory, 2) > 10000
            memory = memory(:, 5000:end);
        end

        % printMap(nextstate, n, n);

        if or(mod(steps, 4) == 0, done)
            [Qnet, flush] = singleTrain(memory, n, layers, Qnet, Tnet);
            flushTime += flush;

            if flush
                verbose = 1;
            end

            if and(mod(flushTime, 150) == 0, flushTime > 0)
                Tnet = Qnet;
                save("Qnet.mat", "Qnet");
                disp("Save Qnet !!");
            end
        end

        state = nextstate;
        totalReward += reward;

        if done
            fprintf("One play round done, total rewars %d after %d steps\n\n", totalReward, steps - startStep);
            trainingRewards += totalReward;
            break
        end
    end

    Eps = minEps + (Eps - minEps) * exp(-decay * Eps);
    fprintf("Total %d steps...(mean reward %f)\n", steps, trainingRewards / t);
    fprintf("Eps to %f\n", Eps);
end