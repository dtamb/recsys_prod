# recsys_prod
Building a multi-stage recommender system.

## Structure
4-part recommendation system:
- Embeddings (feature vectors for users and items)
- Retrieval (sorting 20M data points down to <1000)
- Ranking (by pure relevance)
- Re-ranking (re-weighting to cater for diversity and serendipity)

A highly complex and powerful ranking model may produce superior rankings, but doesn't factor the real latency constraints of a live site with real-time users.
As such, this multi-stage system implements more crude but faster algorithms in the retrieval stage to whittle down millions of items to the hundreds. Then  using some more robust ranking methods are brought out to create a curated ranked list for a given user.
Going beyond relevance, re-ranking is used to show content that is not only relevant, but novel and diverse. 
