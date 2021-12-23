#include <iostream>
#include <InputReader.h>
#include <CF.h>

int main() {
//    const std::string rating = "../data/data/u1.base";
//    const std::string test_rating = "../data/data/u1.test";
//    auto input = InputReader(rating, test_rating);
//    input.parse("train", "\t", false);
//    input.parse("test", "\t", false);
//    CF cf(input);
//    auto items = cf.recommended_items_for_user("1", 20, 0, 10);
//    cf.test_rmse<SP_ROW>(-1, 0);

    const std::string book_train = "../data/Book-Ratings-train.csv";
    const std::string book_test = "../data/Book-Ratings-test.csv";
    auto input = InputReader(book_train, book_test);
    input.parse("train", ";", true);
    input.parse("test", ";", true);
    CF cf(input);
    auto items = cf.recommended_items_for_user("8", "item-based", -1, 0, 10);
//    auto users = cf.recommended_users_for_item("88740", 300);
//    cf.test_rmse<SP_COL>(5.5, -1, 0);
}
