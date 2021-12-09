#pragma once
#include <vector>
#include <eigen3/Eigen/Sparse>

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SP_COL;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SP_ROW;
typedef Eigen::SparseVector<double> SV;
typedef std::vector<std::pair<int, double>> IDX_SCORE_VEC;

class CF {
private:
    double test_u_size = 0.2, test_i_size = 0.2;
    SP_COL UIMAT_Col;  SP_ROW UIMAT_Row;
    static SP_COL get_UIMAT(const std::string & file_path);
    void filter_rare_scoring();
    IDX_SCORE_VEC calculate_simi(const std::string & ui, int idx, double simi_th, int test_start=-1, int mat_test_start=-1);
    IDX_SCORE_VEC naive_kNearest_user(int user, int k, double simi_th);
    IDX_SCORE_VEC naive_kNearest_item(int item, int k, double simi_th);
    SV calculate_weighted_sum(const std::string & ui, const IDX_SCORE_VEC & idx_score);
    double predict_ui_rating(const std::string & ui, int idx, const IDX_SCORE_VEC & idx_score, bool ret_avg=true);
public:
    explicit CF(const std::string & file_path) {
        UIMAT_Col = get_UIMAT(file_path);
        UIMAT_Row = SP_ROW(UIMAT_Col);
//        filter_rare_scoring();
    };
    void test_rmse(const std::string & method);
    IDX_SCORE_VEC recommended_items_for_user(int user_id, int k=30, double simi_th=0.1, int n=5);
    IDX_SCORE_VEC recommended_users_for_item(int item_id, int k=30, double simi_th=0.1, int n=5);
};
