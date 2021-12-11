#include <random>
#include "gtest/gtest.h"
#include "KNN.h"
#include "CF.h"

template<typename SP>
SP make_matrix() {
    SP sp_mat(8, 8);
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

TEST(CFTest, naive_kNearest_user) {
    auto sp_mat = make_matrix<SP_ROW>();
    IDX_SCORE_VEC result = KNN<SP_ROW>::naive_kNearest(sp_mat, 0, 3, 0.5);
    EXPECT_EQ(result.size(), 3);

    EXPECT_EQ(result[0].first, 4);
    EXPECT_NEAR( result[0].second, 0.812587, 1e-6);

    EXPECT_EQ(result[1].first, 2);
    EXPECT_NEAR(result[1].second, 0.784340, 1e-6);

    EXPECT_EQ(result[2].first, 5);
    EXPECT_NEAR(result[2].second, 0.683999, 1e-6);
}

TEST(CFTest, recommended_items_for_user) {
    CF cf(make_matrix<SP_ROW>());
    IDX_SCORE_VEC result = cf.recommended_items_for_user(0, 3, 0.5);
    EXPECT_EQ(result.size(), 3);

    EXPECT_EQ(result[0].first, 6);
    EXPECT_NEAR(result[0].second, 4.43809, 1e-5);

    EXPECT_EQ(result[1].first, 4);
    EXPECT_NEAR(result[1].second, 2.95601, 1e-5);

    EXPECT_EQ(result[2].first, 2);
    EXPECT_NEAR(result[2].second, 2, 1e-5);
}

TEST(CFTest, naive_kNearest_item) {
    auto sp_mat = make_matrix<SP_COL>();
    IDX_SCORE_VEC result = KNN<SP_COL>::naive_kNearest(sp_mat, 0, 3, 0.5);
    EXPECT_EQ(result.size(), 3);

    EXPECT_EQ(result[0].first, 5);
    EXPECT_NEAR( result[0].second, 0.704448, 1e-6);

    EXPECT_EQ(result[1].first, 1);
    EXPECT_NEAR(result[1].second, 0.576002, 1e-6);

    EXPECT_EQ(result[2].first, 3);
    EXPECT_NEAR(result[2].second, 0.551927, 1e-6);
}

TEST(CFTest, recommended_users_for_item) {
    CF cf(make_matrix<SP_ROW>());
    IDX_SCORE_VEC result = cf.recommended_users_for_item(0, 3, 0.5);
    EXPECT_EQ(result.size(), 3);

    EXPECT_EQ(result[0].first, 1);
    EXPECT_NEAR(result[0].second, 7.37131, 1e-5);

    EXPECT_EQ(result[1].first, 5);
    EXPECT_NEAR(result[1].second, 7.30094, 1e-5);

    EXPECT_EQ(result[2].first, 6);
    EXPECT_NEAR(result[2].second, 6.59696, 1e-5);
}

TEST(CFTest, test_rmse) {
    auto mat = make_matrix<SP_ROW>();
    auto result = KNN<SP_ROW>::naive_kNearest(mat, 6, 2, 0.3, 6);
    CF cf(mat);
    double rmse = cf.calculate_rmse<SP_ROW>(6, 0.3, 0.3, 0.3);
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].first, 1);
    EXPECT_NEAR(result[0].second, 0.735627, 1e-6);
    EXPECT_NEAR(rmse, 3.39350, 1e-5);
}
