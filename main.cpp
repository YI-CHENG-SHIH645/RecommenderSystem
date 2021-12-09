#include <iostream>
#include <CF.h>

int main() {
    CF cf("../data/Book_reviews/Book_reviews/BX-Book-Ratings.csv");
    auto items = cf.recommended_items_for_user(54884);
    auto users = cf.recommended_users_for_item(88740);
//    cf.test_rmse("user-based");

//    SP_ROW mat(6, 8);
//    mat.coeffRef(0, 0) = 3;
//    mat.coeffRef(5, 5) = 4;
//    mat.coeffRef(5, 7) = 4;
//    SP_ROW mat_block = mat.bottomRightCorner(3, 4);
//    SV row = mat_block.row(2);
//    std::cout << row << std::endl;
//    for(int i=0; i<row.nonZeros(); ++i) {
//        std::cout << *(row.innerIndexPtr()+i) << std::endl;
//    }
//    SV row = mat.row(5);
//    for(int i=0; i<row.nonZeros(); ++i) {
//        mat.coeffRef(5, *(row.innerIndexPtr() + i)) = 0;
//    }
}
