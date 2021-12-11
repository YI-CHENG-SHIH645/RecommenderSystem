#include <iostream>
#include <CF.h>

int main() {
    CF cf("../data/Book_reviews/Book_reviews/BX-Book-Ratings.csv", false);
    auto items = cf.recommended_items_for_user(54884);
    auto users = cf.recommended_users_for_item(88740);
    double rmse = cf.calculate_rmse<SP_ROW>(30, 0, 0.1, 0.1);
    std::cout << "rmse : " << rmse << std::endl;
//    cf.test_rmse("user-based");
}
