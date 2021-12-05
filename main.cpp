#include <iostream>
#include <CollaborativeFiltering.h>
#include <eigen3/Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SP;

int main() {
    CF<SP> cf("../data/Book_reviews/Book_reviews/BX-Book-Ratings.csv");
    auto items = cf.recommendation_for_a_user(249848);
    auto users = cf.an_item_to_users(249848);
    for(auto item : items)
        std::cout << item.first << " : " << item.second << std::endl;
    std::cout << std::endl << std::endl << std::endl;
    for(auto user : users)
        std::cout << user.first << " : " << user.second << std::endl;

//    SP mat(3, 3);
//    mat.coeffRef(1, 2) = 5;
//    mat.coeffRef(1, 0) = 5;
//    mat.coeffRef(2, 1) = 3;
//    Eigen::SparseVector<double> r1 = mat.row(1);
//    Eigen::SparseVector<double> r2 = mat.row(2);
//    std::cout << r1 << std::endl;
//    std::cout << r2 << std::endl;
//    std::cout << r1.cwiseProduct(r2).sum() << std::endl;
//    std::cout << mat << std::endl;
//    mat.makeCompressed();
//    std::vector<double> vec(r1.innerIndexPtr(), r1.innerIndexPtr()+r1.nonZeros());
//    for(auto v : vec)
//        std::cout << v << std::endl;
//  20 / 249848
    return 0;
}
