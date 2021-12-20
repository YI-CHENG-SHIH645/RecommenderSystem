#include <iostream>
#include <InputReader.h>
#include <CF.h>

int main() {
    const std::string rating = "../data/data/u1.base";
    const std::string test_rating = "../data/data/u1.test";
    auto input = InputReader(rating, test_rating);
    input.parse("train", "\t", false);
    input.parse("test", "\t", false);
    CF cf(input);
//    auto items = cf.recommended_items_for_user(1, 20, 0, 10);
    cf.test_rmse<SP_ROW>(-1, 0);

//    const std::string book = "../data/Book_reviews/Book_reviews/BX-Book-Ratings.csv";
//    CF cf(book, "", ";", true);
//    auto items = cf.recommended_items_for_user(8, 20, 0, 10);
//    auto users = cf.recommended_users_for_item(88740, 300);
}
