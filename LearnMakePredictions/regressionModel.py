from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate regression dataset
X, y = make_regression(n_samples = 100, n_features = 2, noise = 0.1, random_state = 1)
# Fit final model
model = LinearRegression()
model.fit(X, y)

# New instances where we do not know the answer
Xnew, _ = make_regression(n_samples = 3, n_features = 2, noise = 0.1, random_state = 1)
# Make a prediction
ynew = model.predict(Xnew)
# Show the inputs and predicted outputs
for i in range(len(Xnew)) :
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))