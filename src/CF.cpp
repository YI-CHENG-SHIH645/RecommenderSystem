#include <iostream>
#include <unordered_map>
#include "CF.h"
#include "StopWatch.h"
#include "KNN.h"

auto comp_fn = [](const auto & a, const auto & b){
    return a.second == b.second ? a.first < b.first : a.second > b.second;
};

template<typename SP>
double CF::predict_ui_rating(int idx, const IDX_SCORE_VEC & idx_score) const {
    SV vec;
    double ws = 0.0, k = 0.0;  // normalization
    if(SP::IsRowMajor) {
        vec = input.train_data_col().col(idx);
    } else {
        vec = input.train_data_row().row(idx);
    }

    for(auto idx_simi : idx_score) {
        if(vec.coeff(idx_simi.first) == 0) continue;
        ws += vec.coeff(idx_simi.first) * idx_simi.second;
        k += abs(idx_simi.second);
    }

    if(k==0)
        return std::numeric_limits<double>::max();

    return ws/k;
}

IDX_SCORE_VEC CF::recommended_items_for_user(const std::string & user_id, int k, double simi_th, int n) {

    StopWatch sw; double elapsed;
    int user = input.u2i()[user_id];
    auto UIMAT_Row = input.train_data_row();
    auto valid_cols = input.valid_col_idx();
    std::cout << "calculate weighted sum of top k item vectors ... ";  sw.lap();
    IDX_SCORE_VEC sorted_user_simi = KNN<SP_ROW>::naive_kNearest(UIMAT_Row, user, k, simi_th);
    SV weighted_sum(UIMAT_Row.cols());
    SV tgt_row = UIMAT_Row.row(user);
    double rating;
    for(int i=0; i<tgt_row.size(); ++i) {
        if(tgt_row.coeff(i) == 0 && (valid_cols.count(i) || !input.filtered())) {
            rating = predict_ui_rating<SP_ROW>(i, sorted_user_simi);
            if(rating == std::numeric_limits<double>::min())
                continue;
            if(rating != 0)
                weighted_sum.coeffRef(i) = rating;
        }
    }
    size_t num_non_zero = weighted_sum.nonZeros();
    int rec_to_keep = std::min(n, (int)num_non_zero);
    IDX_SCORE_VEC itemID_score(rec_to_keep);
    for(int i=0; i<rec_to_keep; ++i) {
        itemID_score.emplace(itemID_score.begin(), *(weighted_sum.innerIndexPtr()+i), *(weighted_sum.valuePtr()+i));
    }
    itemID_score.resize(rec_to_keep);
//    itemID_score.shrink_to_fit();
    std::sort(itemID_score.begin(), itemID_score.end(), comp_fn);
    elapsed = sw.lap(); std::cout << elapsed << " sec" << std::endl;

    std::cout << "# non-zero score recommendation items : " << num_non_zero << std::endl;
    std::cout << "top n recommendation items for the given user : result_n=" << num_non_zero << std::endl;
    for(auto i : itemID_score)
        std::cout << i.first << " : " << i.second << std::endl;

    return itemID_score;
}

IDX_SCORE_VEC CF::recommended_users_for_item(const std::string & item_id, int k, double simi_th, int n) {

    StopWatch sw; double elapsed;
    int item = input.i2i()[item_id];
    auto UIMAT_Col = input.train_data_col();
    auto valid_rows = input.valid_row_idx();
    std::cout << "calculate weighted sum of top k user vectors ... ";  sw.lap();
    IDX_SCORE_VEC sorted_item_simi = KNN<SP_COL>::naive_kNearest(UIMAT_Col, item, k, simi_th);
//    SV weighted_sum = calculate_weighted_sum("item", sorted_item_simi);
    SV weighted_sum(UIMAT_Col.rows());
    SV tgt_col = UIMAT_Col.col(item);
    double rating;
    for(int i=0; i<tgt_col.size(); ++i) {
        if(tgt_col.coeff(i) == 0 && (valid_rows.count(i) || !input.filtered())) {
            rating = predict_ui_rating<SP_COL>(i, sorted_item_simi);
            if(rating == std::numeric_limits<double>::min())
                continue;
            if(rating != 0)
                weighted_sum.coeffRef(i) = rating;
        }
    }
    size_t num_non_zero = weighted_sum.nonZeros();
    int rec_to_keep = std::min(n, (int)num_non_zero);
    IDX_SCORE_VEC userID_score(rec_to_keep);
    for(int i=0; i<rec_to_keep; ++i) {
        userID_score.emplace(userID_score.begin(), *(weighted_sum.innerIndexPtr()+i), *(weighted_sum.valuePtr()+i));
    }
    userID_score.resize(rec_to_keep);
//    userID_score.shrink_to_fit();
    std::sort(userID_score.begin(), userID_score.end(), comp_fn);
    elapsed = sw.lap(); std::cout << elapsed << " sec" << std::endl;

    std::cout << "# non-zero score recommendation items : " << num_non_zero << std::endl;
    std::cout << "top n recommendation items for the given user : result_n=" << num_non_zero << std::endl;
    for(auto u : userID_score)
        std::cout << u.first << " : " << u.second << std::endl;

    return userID_score;
}

template<typename SP>
double CF::test_rmse(int k, double simi_th) {
    int idx, another_idx, cur_idx = -1, avg_count = 0;
    std::string reader, book;
    double rating, pred_rating, se = 0.0, baseline_se = 0.0;
    IDX_SCORE_VEC simi;
    auto test_data = input.test_data_vec();
    auto reader_to_idx = input.u2i();
    auto book_to_idx = input.i2i();
    SP UIMAT;
    if(SP::IsRowMajor) {
        std::sort(test_data.begin(), test_data.end(), [&reader_to_idx, &book_to_idx](const auto &a, const auto &b) {
            if(reader_to_idx[std::get<0>(a)] == reader_to_idx[std::get<0>(b)])
                return book_to_idx[std::get<1>(a)] < book_to_idx[std::get<1>(b)];
            return reader_to_idx[std::get<0>(a)] < reader_to_idx[std::get<0>(b)];
        });
        UIMAT = input.train_data_row();
    }
    else {
        std::sort(test_data.begin(), test_data.end(), [&book_to_idx, &reader_to_idx](const auto &a, const auto &b) {
            if(book_to_idx[std::get<1>(a)] == book_to_idx[std::get<1>(b)])
                return reader_to_idx[std::get<0>(a)] < reader_to_idx[std::get<0>(b)];
            return book_to_idx[std::get<1>(a)] < book_to_idx[std::get<1>(b)];
        });
        UIMAT = input.train_data_col();
    }
    std::cout << "test rmse sorting done" << std::endl;
    int count = 0;
    for(auto & r : test_data) {
        count++;
        if(count % 5000 == 0) std::cout << count << std::endl;
        std::tie(reader, book, rating) = r;
        if(SP::IsRowMajor) {
            idx = reader_to_idx.count(reader) ? reader_to_idx[reader] : -1;
            another_idx = book_to_idx.count(book) ? book_to_idx[book] : -1;
        } else {
            idx = book_to_idx.count(book) ? book_to_idx[book] : -1;
            another_idx = reader_to_idx.count(reader) ? reader_to_idx[reader] : -1;
        }

        if(idx == -1 || another_idx == -1) {
            pred_rating = 5.5;
            ++avg_count;
        } else if(idx != cur_idx) {
            cur_idx = idx;
            simi = KNN<SP>::naive_kNearest(UIMAT, idx, k, simi_th);
            pred_rating = predict_ui_rating<SP>(another_idx, simi);
            if(pred_rating == std::numeric_limits<double>::max()) {
                pred_rating = 5.5;
                ++avg_count;
            }
        } else {
            pred_rating = predict_ui_rating<SP>(another_idx, simi);
            if(pred_rating == std::numeric_limits<double>::max()) {
                pred_rating = 5.5;
                ++avg_count;
            }
        }
        se += pow(rating-pred_rating, 2);
        baseline_se += pow(rating-5.5, 2);
    }
    double rmse = sqrt(se/(double)test_data.size());
    std::cout << "avg ratio : " << avg_count/(double)input.test_data_vec().size() << std::endl;
    std::cout << "rmse : " << rmse << std::endl;
    std::cout << "baseline rmse : " << sqrt(baseline_se/(double)test_data.size()) << std::endl;

    return rmse;
}

template double CF::predict_ui_rating<SP_COL>(int idx, const IDX_SCORE_VEC &idx_score) const;
template double CF::predict_ui_rating<SP_ROW>(int idx, const IDX_SCORE_VEC &idx_score) const;

template double CF::test_rmse<SP_COL>(int k, double simi_th);
template double CF::test_rmse<SP_ROW>(int k, double simi_th);
