# Key Decisions
Key assumptions and decisions made throughout the project. 

## Retrieval
**Why Alternating Least Squares (ALS)?**
Can handle large datasets with Spark support for parallelisation. Strong with implicit and sparse datasets.

## Feature Engineering
**Why PCA instead of raw genome tags?**
Reduce dimensions of genome tags (1128 --> ~45). Faster computation and less noise.

**Why heuristic weighting of user tags?**
Provides a simpler and faster baseline. Ridge regression planned as future improvement.

## Evaluation
**Why NDCG, MAP, Precision and Recall over MAE or RMSE?**
Rating prediction error doesn't translate as well to user preferences  on a ranked list. 

NDCG, in particular, better evaluates ranking recommendations and their order.