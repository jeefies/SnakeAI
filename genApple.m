function M = genApple(mmap, n)
    M = randi(n, 1, 2);
    while mmap(M(1), M(2)) > 0
        % fprintf("Exist at (%d, %d) regenerate...\n", M(1), M(2));
        M = randi(n, 1, 2);
    end
    % fprintf("New Apple at (%d, %d)\n", M(1), M(2));
end