#pragma once
#include <utility>
#include <vector>
#include <unordered_map>
#include <set>
#include <eigen3/Eigen/Sparse>
#include <KNN.h>
#include <InputReader.h>


class CF {
private:
    InputReader & input;

    template<typename SP>
    [[nodiscard]] double predict_ui_rating(int idx, const IDX_SCORE_VEC & idx_score) const;
public:
    explicit CF(InputReader & input): input(input) {};

    template<typename SP>
    double test_rmse(int k, double simi_th);

    IDX_SCORE_VEC recommended_items_for_user(const std::string & user_id, int k=20, double simi_th=0, int n=10);
    IDX_SCORE_VEC recommended_users_for_item(const std::string & item_id, int k=20, double simi_th=0, int n=10);
};
