function targetValue = apply_saturation(targetValue)
    % saturation between 0.34 and -0.34
    for j = 1:length(targetValue)
        if targetValue(j) >= 0.34
           targetValue(j) = 0.34;
        elseif targetValue(j) <= -0.34
           targetValue(j) = -0.34;
        end
    end
end
