### Linear Regression & Classification

A generic notebook describing the methods can be found [here](Scouting/Linear%20Models%20%20for%20Regression%20%26%20Classification.ipynb).

|        | [SparseRegression.jl](Scouting/Sparse%20%Regression.ipynb) |  [MultivariateStats.jl](Scouting/MultivariateStats.ipynb) | [OnlineStats.jl](Scouting/OnlineStats.ipynb) |
| ------------- |:-------------:|:-------------:|:-:|
| Package works | yes | yes | yes |
| Deprecations warnings      | No | No | No |
| Compatible with JuliaDB | If transformed into matrix | If transformed into matrix | If transformed into matrix |
| Contains documentation | yes, but not great | very good | Yes, mostly very good |
| Simplicity | good | good | High |



### Decision Trees

|                       | [DecisionTrees.jl](Scouting/DecisionTree.ipynb) |  [ScikitLearn.jl](Scouting/ScikitLearn.jl.ipynb) |
| :-: | :-: | :-: |
| Packages works            | yes                               | Yes |
| Deprecation warnings      | None                              | Some      |
| Compatible with JuliaDB   | If tables are converted to arrays | Yes (transformation of tables to arrays required) |
| Contains Documetation     | No, but many examples             | Yes (very good!) |
| Simplicity                | Good, like sklearn                | good |

### Utilities

- _MLPreprocessing_ is used to do simple scaling/normalising
- _MLLabelUtils_ provides functions to modify the labels to be compatible with whatever the algorithm requires. For instance transform into categorical, booleans, from text to number etc
- MLBase provides function for label encoding, classification from model scores, performance evaluation (ROC, F1 etc),
cross-validation (Kfold, stratified Kfold, subsmapling), and grid search hyperparameter tuning.



|                       | [MLPreprocessing.jl](Scouting/MLPreprocessing.ipynb) | [MLLabelUtils.jl](Scouting/MLLabelUtils.ipynb) | [MLBase.jl](Scouting/MLBase.jl.ipynb)|
| :- | :- |
| Packages works            | yes | yes || yes|
| Deprecation warnings      | None | No ||No|
| Compatible with JuliaDB   | If tables are converted to arrays or dataframes | If tables transformed into arrays |If tables transformed into arrays|
| Contains Documetation     | No, but sufficient examples | yes ||yes|
| Simplicity                | Fair | good ||good|


