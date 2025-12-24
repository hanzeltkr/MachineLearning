from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs

# Generate 2d classification dataset
X, y = make_blobs(n_samples = 100, centers = 2, n_features = 2, random_state = 1)
# Fit final model
model = LogisticRegression()
model.fit(X, y)

# New instances where we do not know the answer
Xnew, _ = make_blobs(n_samples = 3, centers = 2, n_features = 2, random_state = 1)
# Make a prediction
ynew = model.predict(Xnew)
# Show the inputs and predicted outputs
for i in range(len(Xnew)) :
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

# Probability prediction
ynew = model.predict_proba(Xnew)
for i in range(len(Xnew)) :
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

