#pragma once
#include <vector>
#include <unordered_map>
#include <set>
#include <eigen3/Eigen/Sparse>
#include <KNN.h>


class CF {
private:
    SP_COL UIMAT_Col;  SP_ROW UIMAT_Row;
    std::set<long> valid_rows, valid_cols;
    std::unordered_map<std::string, int> book_to_idx;
    std::vector<std::tuple<int, std::string, double>> test_data;
    bool filter = false;
    SP_COL get_UIMAT(const std::string & file_path, const std::string & test_file_path="",
                     const char* dlm=",", bool skip_header=false);
    void filter_rare_scoring(int user_th, int item_th);

    template<typename SP>
    double predict_ui_rating(int idx, const IDX_SCORE_VEC & idx_score);
public:
    explicit CF(const std::string & file_path,
                const std::string & test_file_path="",
                const char* dlm=",",
                bool skip_header=false,
                int user_th=-1, int item_th=-1) {
        UIMAT_Col = get_UIMAT(file_path, test_file_path, dlm, skip_header);
        UIMAT_Row = SP_ROW(UIMAT_Col);
        if(user_th > -1 || item_th > -1) {
            filter_rare_scoring(user_th, item_th);
            filter = true;
        }
    };
    explicit CF(const SP_COL & MAT) {
        UIMAT_Col = SP_COL(MAT);
        UIMAT_Row = SP_ROW(MAT);
//        filter_rare_scoring();
    }

    template<typename SP>
    double calculate_rmse(int k, double simi_th, double test_u_size, double test_i_size);

    template<typename SP>
    double test_rmse(int k, double simi_th);

    IDX_SCORE_VEC recommended_items_for_user(int user_id, int k=30, double simi_th=0.1, int n=5);
    IDX_SCORE_VEC recommended_users_for_item(int item_id, int k=30, double simi_th=0.1, int n=5);
};
