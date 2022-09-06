# automate_LinearRegression

- ### This package is specific to build a Linear Regeression model on the dataset passed as the parameter with its features as well as label to be regressed against. And as such, it is presumed that the analysis and the feature engineering has already been done on the dataset.
    
- ### This package will standardize the data via StandardScaler() so that all features can be on the same scale and obviously so that the model optimization could increase.

- ### Accuracy of the model built by this package is computed using `adjusted R-squared`.

- ### User will also have the flexibility of building the model with regularization modes viz. `Lasso (L1`, `Ridge (L2)` and `ElasticNet` and to compare their accuracies accordingly.

## An Example of How to Use: 

```python
from automate_LinearRegression import automate_linReg

df        # the dataset you want to build the model of
features  # df's features
label     # df's label

linear_model = automate_linReg(df, features, label)
linear_model.split(testSize = 0.15) # first and foremost step is to split the features and label into train and test subdata
linear_model.build()
print(linear_model.accuracy()) # will print the accuracy computed using adjusted R-squared
```


```python
# should the user desires to build the model using regularization

linear_model.buildLasso()                # or linear_model.buildRidge() or linear_model.buildElasticNet()
print(linear_model.accuracy(mode="L1"))  # or mode="L2" or mode="Elastic"
```


```python
linear_model.predict(test_input, mode="L1") # to yield the prediction outcome
```
