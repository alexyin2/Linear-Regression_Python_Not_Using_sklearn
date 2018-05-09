# Learning Linear Regression in Python but Not Using sklearn

### Warning: Linear Algebra and Calculus are highly-needed for learning linear regression.

* In this repository, I'll try my best to avoid using sklearn in Python since my goal is to understand how the mathematics work in linear regression.
* But I'll use packages like Numpy, matplotlib, and sometimes Pandas to help me skip trivial obstacles.
* I'll introduce how these packages help in learning regression model:
1. **Numpy**: 
   - Numpy packages can help solve linear algebra questions really fast.
   - In np.array, calculations are element wise, and this is very useful in doing mathematics.
   - Following are some useful codes for calculating linear algebra by using Numpy:
```
# Inverse Matrix
np.linalg.inv()
# Extract a diagonal or construct a diagonal array.
np.diag()
# Calculate the outer product.
np.outer()
# Calculate the inner product.
np.inner()
```

2. **Pandas**: 
   - Pandas can help importing data and do some summary statistics

3. **matplotlib.pyplot**: 
   - Helps for data visualization

* There are some necessary mathematicals we would have to know:
1. Least Square Error
2. Maximum Likelihood
* This practice is based on [LazyProgrammer.me](https://github.com/lazyprogrammer)

### New Question: Least Square Vs Gradient Descent
After I've tried to code the algorithms by myself, a question popped up in my mind. 

What's the difference between using Least Square and using Gradient Descent

We know that if using Least square, there should be an inverse of X matrix, which implies that each variables should not be a linear combination of other variables.

So perhaps in using R or Python, the packages all use Gradient Descent?
