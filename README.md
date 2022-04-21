# GMM-VSG
A virtual sample generation algorithm based on Gaussian Mixture Model (GMM-VSG) 

## Requirements<br>
* Python3
* Numpy
* Pandas
* Sklearn
<br>

## Run
`python ./code/demo.py --file_path "../data/23items.csv"
                --feature_start_idx 2
                --feature_end_idx 7
                --n_n_components [i for i in range(1, 24, 1)]
                --target_name "akronAbrasion"`
* Necessary parameters
    | parameter | type | description |  
    | :--- | :---: | :---: |  
    | file_path | str | File path of original data |  
    | feature_start_idx | int | Start column of original data feature space |
    | feature_end_idx | int | End column of original data feature space |
    | n_components | list | Search list based on grid search optimization model |
    | target_name | str | Original data target value name |
* Default parameters
    | parameter | type | default value | description |  
    | :--- | :---: | :---: | :---: |  
    | covariance_type | str | "full" | Covariance form of GMM |  
    | optimizer | str | "aic" | Model selection method |
    | result_retain | int  | 4 | Number of decimal places reserved for the result |
    | seed | int | 0 | Random seed in random sampling |
    | new_samples | int | 200 | Number of virtual samples ready for enhancement |
