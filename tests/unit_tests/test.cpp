#include <random>
#include "gtest/gtest.h"
#include "KNN.h"

SP_ROW make_matrix() {
    SP_ROW sp_mat(8, 8);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.emplace_back(Eigen::Triplet<double>(0, 0, 4));
    triplets.emplace_back(Eigen::Triplet<double>(0, 1, 10));
    triplets.emplace_back(Eigen::Triplet<double>(0, 3, 1));
    triplets.emplace_back(Eigen::Triplet<double>(0, 5, 5));
    triplets.emplace_back(Eigen::Triplet<double>(0, 7, 4));

    triplets.emplace_back(Eigen::Triplet<double>(1, 1, 6));
    triplets.emplace_back(Eigen::Triplet<double>(1, 2, 4));
    triplets.emplace_back(Eigen::Triplet<double>(1, 3, 8));
    triplets.emplace_back(Eigen::Triplet<double>(1, 5, 8));
    triplets.emplace_back(Eigen::Triplet<double>(1, 6, 7));

    triplets.emplace_back(Eigen::Triplet<double>(2, 0, 8));
    triplets.emplace_back(Eigen::Triplet<double>(2, 1, 7));
    triplets.emplace_back(Eigen::Triplet<double>(2, 4, 2));
    triplets.emplace_back(Eigen::Triplet<double>(2, 5, 8));
    triplets.emplace_back(Eigen::Triplet<double>(2, 6, 8));
    triplets.emplace_back(Eigen::Triplet<double>(2, 7, 5));

    triplets.emplace_back(Eigen::Triplet<double>(3, 0, 5));
    triplets.emplace_back(Eigen::Triplet<double>(3, 2, 6));
    triplets.emplace_back(Eigen::Triplet<double>(3, 4, 2));
    triplets.emplace_back(Eigen::Triplet<double>(3, 5, 1));
    triplets.emplace_back(Eigen::Triplet<double>(3, 6, 3));
    triplets.emplace_back(Eigen::Triplet<double>(3, 7, 4));

    triplets.emplace_back(Eigen::Triplet<double>(4, 0, 9));
    triplets.emplace_back(Eigen::Triplet<double>(4, 1, 9));
    triplets.emplace_back(Eigen::Triplet<double>(4, 3, 7));
    triplets.emplace_back(Eigen::Triplet<double>(4, 4, 3));
    triplets.emplace_back(Eigen::Triplet<double>(4, 5, 10));
    triplets.emplace_back(Eigen::Triplet<double>(4, 6, 1));

    triplets.emplace_back(Eigen::Triplet<double>(5, 1, 4));
    triplets.emplace_back(Eigen::Triplet<double>(5, 2, 2));
    triplets.emplace_back(Eigen::Triplet<double>(5, 4, 4));
    triplets.emplace_back(Eigen::Triplet<double>(5, 5, 10));
    triplets.emplace_back(Eigen::Triplet<double>(5, 7, 4));

    triplets.emplace_back(Eigen::Triplet<double>(6, 1, 8));
    triplets.emplace_back(Eigen::Triplet<double>(6, 2, 11));
    triplets.emplace_back(Eigen::Triplet<double>(6, 3, 11));
    triplets.emplace_back(Eigen::Triplet<double>(6, 5, 2));
    triplets.emplace_back(Eigen::Triplet<double>(6, 6, 3));
    triplets.emplace_back(Eigen::Triplet<double>(6, 7, 7));

    triplets.emplace_back(Eigen::Triplet<double>(7, 0, 11));
    triplets.emplace_back(Eigen::Triplet<double>(7, 1, 1));
    triplets.emplace_back(Eigen::Triplet<double>(7, 2, 7));
    triplets.emplace_back(Eigen::Triplet<double>(7, 3, 10));
    triplets.emplace_back(Eigen::Triplet<double>(7, 5, 6));
    triplets.emplace_back(Eigen::Triplet<double>(7, 7, 1));

    sp_mat.setFromTriplets(triplets.begin(), triplets.end());

    return sp_mat;
}

TEST(CFTest, naive_kNearest) {
    SP_ROW sp_mat = make_matrix();
    IDX_SCORE_VEC result = KNN<SP_ROW>::naive_kNearest(sp_mat, 0, 3, 0.5);
    EXPECT_EQ(result.size(), 3);

    EXPECT_EQ(result[0].first, 4);
    EXPECT_NEAR( result[0].second, 0.812587, 1e-6);

    EXPECT_EQ(result[1].first, 2);
    EXPECT_NEAR(result[1].second, 0.784340, 1e-6);

    EXPECT_EQ(result[2].first, 5);
    EXPECT_NEAR(result[2].second, 0.683999, 1e-6);

}
