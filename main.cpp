#include <iostream>
#include <CF.h>

int main() {
    const std::string rating = "../data/data/u1.base";
    const std::string test_rating = "../data/data/u1.test";
    CF cf(rating, test_rating, "\t", false);
//    auto items = cf.recommended_items_for_user(1, 20, 0, 10);
    cf.test_rmse<SP_ROW>(-1, 0);

//    const std::string book = "../data/Book_reviews/Book_reviews/BX-Book-Ratings.csv";
//    CF cf(book, "", ";", true);
//    auto items = cf.recommended_items_for_user(8, 20, 0, 10);
//    auto users = cf.recommended_users_for_item(88740, 300);
//    cf.calculate_rmse<SP_ROW>(-1, 0, 0.3, 0.3);

//    SP_ROW sp_mat(4, 6);
//    sp_mat.coeffRef(0, 0) = 1;
//    sp_mat.coeffRef(0, 1) = 2;
//    sp_mat.coeffRef(1, 1) = 3;
//    sp_mat.coeffRef(1, 3) = 4;
//    sp_mat.coeffRef(2, 2) = 5;
//    sp_mat.coeffRef(2, 3) = 6;
//    sp_mat.coeffRef(2, 4) = 7;
//    sp_mat.coeffRef(3, 5) = 8;
//    sp_mat.makeCompressed();
//    for(long i=0; i<sp_mat.nonZeros(); ++i)
//        std::cout << *(sp_mat.valuePtr()+i) << " ";
//    std::cout << std::endl;
//    for(long i=0; i<sp_mat.nonZeros(); ++i)
//        std::cout << *(sp_mat.innerIndexPtr()+i) << " ";
//    std::cout << std::endl;
//    for(long i=0; i<sp_mat.rows()+1; ++i)
//        std::cout << *(sp_mat.outerIndexPtr()+i) << " ";
//    std::cout << std::endl << std::endl;
//    SP_COL aaaa = sp_mat.transpose();
//    std::cout << aaaa.IsRowMajor << std::endl;
//    std::cout << aaaa << std::endl;
//    std::cout << sp_mat << std::endl;
}
