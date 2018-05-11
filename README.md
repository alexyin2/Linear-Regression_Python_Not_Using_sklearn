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

* There are some necessary mathematicals or algorithms we would have to know:
1. Least Square Error  
2. Maximum Likelihood  
3. Gradient Descent  
* This practice is based on [LazyProgrammer.me](https://github.com/lazyprogrammer)


***
## New Question: Least Square Vs Gradient Descent
* After I've tried to code the algorithms by myself, Several questions popped up in my mind.   
* This is the answer that I've finally got from searching and reading.  
* Feel free to rewrite them if I'm wrong!!  

### 1. What's the difference between using Least Square and using Gradient Descent?

   **_Ans_:**  
   We know that if using Least square, there should be an inverse of X matrix, which implies that each variables should not be a linear combination of other variables.
   Besides, we need to avoid the dummy variable trap, since it will cause the problem of calculating X inverse matrix.  
   
   In using Gradient Descent, we don't need to calculate the inverse of X matrix, and we also don't need to worry about the problem of dummy varialbe trap.  
   But we'll have to decide our learning rate, which may effect the speed of finding the solutions.   

### 2. When using R or Python, the packages use Gradient Descent instead of OLS?
   
   **_Ans_:**  
    
   The answer is YES. There are several reasons why we often use Gradient Descent in machine learning but we seldom use OLS.
   
   **Computational Speed**:
   
      In machine learning, we often have really large data and high dimensions of variables. 
      As a result, calculating the inverse matrix of X may cause a lot of time when using Gradient Descent may be more efficient.
      
   **Mathematic Problems**: 
      
      In some cases, OLS doesn't work. For example, when running L1 Regularization, also known as LASSO Regression, we can't use OLS to get our solution. So using Gradient Descent may be more safe to apply on different situations.
      
      Besides, when using OLS, there is a restriction that N > n. Here N means the number of samples, and n means the number of variables. But in Gradient Descent, there is no such restriction.

