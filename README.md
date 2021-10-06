# Estimating Error Covariance with KalmanNet

## Running the linear model

1. in `src/filing_paths.py` uncomment the line `path_model = path_model_Lin` to select the proper path for the model
2. in `src/mainLinear.py` Set the proper parameters at the beginning of `main()` (more info on how to set them can be found there)
3. run `python src/mainLinear`


## Running the Lorenz attractor
1. in `src/filing_paths.py` uncomment the line `path_model = path_model_Lor` to select the proper path for the model
2. run `python src/mainCovLorenz.py` (no parameters need to be set here)


