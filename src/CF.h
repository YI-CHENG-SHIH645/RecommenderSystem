#pragma once
#include <vector>
#include <eigen3/Eigen/Sparse>
#include <KNN.h>

SP_COL get_UIMAT(const std::string & file_path);

class CF {
private:
    SP_COL UIMAT_Col;  SP_ROW UIMAT_Row;
    double test_u_size = 0.2, test_i_size = 0.2;
    void filter_rare_scoring();
public:
    explicit CF(const std::string & file_path) {
        UIMAT_Col = get_UIMAT(file_path);
        UIMAT_Row = SP_ROW(UIMAT_Col);
        filter_rare_scoring();
    };

    explicit CF(const SP_COL & MAT) {
        UIMAT_Col = MAT;
        UIMAT_Row = SP_ROW(MAT);
//        filter_rare_scoring();
    }

//    void test_rmse(const std::string & method);
    double predict_ui_rating(const std::string & ui, int idx, const IDX_SCORE_VEC & idx_score, bool ret_avg=true);
    IDX_SCORE_VEC recommended_items_for_user(int user_id, int k=30, double simi_th=0.1, int n=5);
    IDX_SCORE_VEC recommended_users_for_item(int item_id, int k=30, double simi_th=0.1, int n=5);
};
