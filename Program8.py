gridSize = [6, 6];
startPos = [1, 1];
goalPos = [6, 6];
obstacles = [2 2; 2 3; 3 3; 4 5; 5 2];

gridWorld = ones(gridSize);
for i = 1:size(obstacles,1)
gridWorld(obstacles(i,1), obstacles(i,2)) = 2;
end
gridWorld(goalPos(1), goalPos(2)) = 3;

figure;
imagesc(gridWorld);
colormap([1 1 1; 0 0 0; 0 1 0]);
axis equal tight;
xticks(1:gridSize(2));
yticks(1:gridSize(1));
grid on;
xlabel('Column');
ylabel('Row');
title('Custom 2D Grid World');
hold on;
plot(startPos(2), startPos(1), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
text(startPos(2)+0.1, startPos(1), 'Start', 'Color', 'r', 'FontSize', 10);
text(goalPos(2)+0.1, goalPos(1), 'Goal', 'Color', 'g', 'FontSize', 10);

agentPos = startPos;
path = agentPos;

while ~isequal(agentPos, goalPos)
pause(0.2);
moves = [-1 0; 1 0; 0 -1; 0 1];
nextMoves = agentPos + moves;
valid = [];
for i = 1:4
r = nextMoves(i,1);
c = nextMoves(i,2);
if r >= 1 && r <= gridSize(1) && c >= 1 && c <= gridSize(2)
if gridWorld(r,c) ~= 2
valid = [valid; r c];
end
end
end
if isempty(valid)
disp('Agent is stuck! No valid moves.');
break;
end
nextPos = valid(randi(size(valid,1)), :);
agentPos = nextPos;
path = [path; agentPos];
end

plot(agentPos(2), agentPos(1), 'bs', 'MarkerSize', 10, 'LineWidth', 1.5);
drawnow;
plot(path(:,2), path(:,1), 'r--', 'LineWidth', 2);
title('Agent Path to Goal');

fprintf('\nAgent Path from Start to Goal:\n');
for i = 1:size(path,1)
fprintf('Step %2d: Row = %d, Column = %d\n', i, path(i,1), path(i,2));
end
