#pragma once
#include <vector>
#include <eigen3/Eigen/Sparse>
#include <KNN.h>

SP_COL get_UIMAT(const std::string & file_path);

class CF {
private:
    SP_COL UIMAT_Col;  SP_ROW UIMAT_Row;
    void filter_rare_scoring();

    template<typename SP>
    double predict_ui_rating(int idx, const IDX_SCORE_VEC & idx_score, bool ret_avg=false);
public:
    explicit CF(const std::string & file_path, bool filter_rare=false) {
        UIMAT_Col = get_UIMAT(file_path);
        UIMAT_Row = SP_ROW(UIMAT_Col);
        if(filter_rare)
            filter_rare_scoring();
    };
    explicit CF(const SP_COL & MAT) {
        UIMAT_Col = SP_COL(MAT);
        UIMAT_Row = SP_ROW(MAT);
//        filter_rare_scoring();
    }

    template<typename SP>
    double calculate_rmse(int k, double simi_th, double test_u_size, double test_i_size);
    IDX_SCORE_VEC recommended_items_for_user(int user_id, int k=30, double simi_th=0.1, int n=5);
    IDX_SCORE_VEC recommended_users_for_item(int item_id, int k=30, double simi_th=0.1, int n=5);
};
