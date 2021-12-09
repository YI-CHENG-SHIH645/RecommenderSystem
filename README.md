# Recommender System

Problem To Solve
=================
Implement User-based / Item-based top-N recommendation Collaborative Filtering.
  - User-based CF : 
    ```
    Given a pair (user, item), find top-k users similar(cos similarity) to the given user 
    from the user-item rating matrix, then use similarities to do weighted sum
    if the rating is non-zero, then we get the user-item rating prediction.
    ```
  - Item-based CF:
    ```
    Given a pair (user, item), find top-k items similar(cos similarity) to the give item 
    from the user-item rating matrix, then use similarities to do weighted sum
    if the rating is non-zero, then we get the user-item rating prediction.
    ```

I need a KNN algorithm that's sufficient to find top k similar neighbor
the straightforward approach is to do the full search, 
it might be ok since the matrix would be a sparse matrix
if not so, consider some approx. approach like LSH to find near optimal solution

Finally, we return top n ratings that are not yet rated by the given user u / item i,
as the recommendation.


Prospective users
==================
Anyone who wants to get recommendation for users, the item can
be whatever it might be, like movie, book, music...


System architecture
===================
    LSH(user_id, input_file, k): return user_ids (k nearest neighbor)

    recommend(user_ids, input_file, output_file, k): write recommendations to the output_file
        aggregate result from LSH with user-item matrix

user_ids: It can receive multiple users at once.

input_file: each row contains user_id, item_id, rating.

output_file: write the recommendations for each users to here.


API description
================
python API:

    recommend(user_ids: list[int], input_file: str, output_file: str, k: int)

k: 'k' nearest neighbor


Engineering infrastructure
===========================

* build system
    * make

* testing framework
    * c++: GoogleTest
    * Python: pytest


Schedule
=========
week1:
prototype on C++

week2:
prototype on C++
implement LSH

week3:
implement LSH

week4:
implement LSH

week5:
aggregate result from LSH with user-item matrix to recommend

week6:
aggregate result from LSH with user-item matrix to recommend
python11 python API

week7:
prepare presentation

week8:
prepare presentation


References
==========
https://en.wikipedia.org/wiki/Collaborative_filtering

https://github.com/bowbowbow/CollaborativeFiltering

https://github.com/cchatzis/Nearest-Neighbour-LSH
