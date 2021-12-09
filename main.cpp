#include <iostream>
#include <CF.h>

int main() {
    CF cf("../data/Book_reviews/Book_reviews/BX-Book-Ratings.csv");
    auto items = cf.recommended_items_for_user(54884);
    auto users = cf.recommended_users_for_item(88740);
    cf.test_rmse("user-based");
}
