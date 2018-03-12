## List of useful tips

### For non tree based models
#### Preprocessing
 - There is multiple kind of features for preprocessing:
    - Categorical: A features with distinct categories (ex: Gender)
    - Ordinal: Similar to categorical features but with an order (ex: Education -> undergraduate, bachelor, master, doctoral)
    - Continuous: Can take a wide range of values (ex distance in km)
 - Log transformation with `np.log(1 + x)` (especially helps neural networks)
 - Raising to the power < 1: `np.sqrt(x + 2 / 3)` (especially helps neural networks)