from sklearn.metrics import davies_bouldin_score

X = [[1, 2], [1,4], [1, 0], [10, 2], [10, 4], [10, 0]]
labels = [0, 0, 0, 1, 1, 1]
score = davies_bouldin_score(X, labels)
print(score)