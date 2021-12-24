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

ID_SCORE_VEC CF::recommended_items_for_user(const std::string & user_id,
                                             const std::string & based,
                                             int k, double simi_th, int n,
                                             bool keep_nonzero_topk) {
    StopWatch sw; double elapsed;
    int user = input.usr2idx()[user_id];
    auto UIMAT_Row = input.train_data_row();
    auto UIMAT_Col = input.train_data_col();
    auto valid_idx = based == "user-based" ? input.valid_col_idx() : input.valid_row_idx();
    std::cout << "calculate weighted sum of top k item vectors ... ";  sw.lap();
    IDX_SCORE_VEC sorted_simi;
    if(based == "user-based" && !keep_nonzero_topk) {
        sorted_simi = KNN<SP_ROW>::naive_kNearest(UIMAT_Row, user, -1, k, simi_th);
    }
    SV weighted_sum(UIMAT_Row.cols());
    SV tgt_row = UIMAT_Row.row(user);
    double rating;
    for(int i=0; i<tgt_row.size(); ++i) {
        if(tgt_row.coeff(i) == 0 && (valid_idx.count(i) || !input.filtered())) {
            if(based == "user-based" && !keep_nonzero_topk) {
                rating = predict_ui_rating<SP_ROW>(i, sorted_simi);
            } else if(based == "user-based" && keep_nonzero_topk) {
                sorted_simi = KNN<SP_ROW>::naive_kNearest(UIMAT_Row, user, i, k, simi_th);
                rating = predict_ui_rating<SP_ROW>(i, sorted_simi);
            } else if(based == "item-based" && !keep_nonzero_topk) {
                sorted_simi = KNN<SP_COL>::naive_kNearest(UIMAT_Col, i, -1, k, simi_th);
                rating = predict_ui_rating<SP_COL>(user, sorted_simi);
            } else {
                sorted_simi = KNN<SP_COL>::naive_kNearest(UIMAT_Col, i, user, k, simi_th);
                rating = predict_ui_rating<SP_COL>(user, sorted_simi);
            }
            if(rating == std::numeric_limits<double>::max())
                continue;
            if(rating != 0)
                weighted_sum.coeffRef(i) = rating;
        }
    }
    size_t num_non_zero = weighted_sum.nonZeros();
    int rec_to_keep = std::min(n, (int)num_non_zero);
    ID_SCORE_VEC itemID_score(rec_to_keep);
    for(int i=0; i<rec_to_keep; ++i) {
        itemID_score.emplace(itemID_score.begin(),
                             input.idx2item()[*(weighted_sum.innerIndexPtr()+i)],
                             *(weighted_sum.valuePtr()+i));
    }
    itemID_score.resize(rec_to_keep);
//    itemID_score.shrink_to_fit();
    std::sort(itemID_score.begin(), itemID_score.end(), comp_fn);
    elapsed = sw.lap(); std::cout << elapsed << " sec" << std::endl;

    std::cout << "# non-zero score recommendation items : " << num_non_zero << std::endl;
    std::cout << "top n recommendation items for the given user : result_n=" << num_non_zero << std::endl;
    for(const auto & i : itemID_score)
        std::cout << i.first << " : " << i.second << std::endl;

    return itemID_score;
}

ID_SCORE_VEC CF::recommended_users_for_item(const std::string & item_id,
                                             const std::string & based,
                                             int k, double simi_th, int n,
                                             bool keep_nonzero_topk) {
    StopWatch sw; double elapsed;
    int item = input.item2idx()[item_id];
    auto UIMAT_Col = input.train_data_col();
    auto UIMAT_Row = input.train_data_row();
    auto valid_idx = based == "user-based" ? input.valid_col_idx() : input.valid_row_idx();
    std::cout << "calculate weighted sum of top k user vectors ... ";  sw.lap();
    IDX_SCORE_VEC sorted_simi;
    if(based == "item-based" && !keep_nonzero_topk) {
        sorted_simi = KNN<SP_COL>::naive_kNearest(UIMAT_Col, item, -1, k, simi_th);
    }
    SV weighted_sum(UIMAT_Col.rows());
    SV tgt_col = UIMAT_Col.col(item);
    double rating;
    for(int i=0; i<tgt_col.size(); ++i) {
        if(tgt_col.coeff(i) == 0 && (valid_idx.count(i) || !input.filtered())) {
            if(based == "item-based" && !keep_nonzero_topk) {
                rating = predict_ui_rating<SP_COL>(i, sorted_simi);
            } else if(based == "item-based" && keep_nonzero_topk) {
                sorted_simi = KNN<SP_COL>::naive_kNearest(UIMAT_Col, item, i, k, simi_th);
                rating = predict_ui_rating<SP_COL>(i, sorted_simi);
            } else if(based == "user-based" && !keep_nonzero_topk) {
                sorted_simi = KNN<SP_ROW>::naive_kNearest(UIMAT_Row, i, -1, k, simi_th);
                rating = predict_ui_rating<SP_ROW>(item, sorted_simi);
            } else {
                sorted_simi = KNN<SP_ROW>::naive_kNearest(UIMAT_Row, i, item, k, simi_th);
                rating = predict_ui_rating<SP_ROW>(item, sorted_simi);
            }
            if(rating == std::numeric_limits<double>::max())
                continue;
            if(rating != 0)
                weighted_sum.coeffRef(i) = rating;
        }
    }
    size_t num_non_zero = weighted_sum.nonZeros();
    int rec_to_keep = std::min(n, (int)num_non_zero);
    ID_SCORE_VEC userID_score(rec_to_keep);
    for(int i=0; i<rec_to_keep; ++i) {
        userID_score.emplace(userID_score.begin(),
                             input.idx2usr()[*(weighted_sum.innerIndexPtr()+i)],
                             *(weighted_sum.valuePtr()+i));
    }
    userID_score.resize(rec_to_keep);
//    userID_score.shrink_to_fit();
    std::sort(userID_score.begin(), userID_score.end(), comp_fn);
    elapsed = sw.lap(); std::cout << elapsed << " sec" << std::endl;

    std::cout << "# non-zero score recommendation items : " << num_non_zero << std::endl;
    std::cout << "top n recommendation items for the given user : result_n=" << num_non_zero << std::endl;
    for(const auto & u : userID_score)
        std::cout << u.first << " : " << u.second << std::endl;

    return userID_score;
}

template<typename SP>
double CF::test_rmse(double avg_value, int k, double simi_th, bool keep_nonzero_topk) {
    avg_value = avg_value == -1 ? input.train_data_col().sum()/(double)input.train_data_col().nonZeros() : avg_value;
    int idx, another_idx, avg_count = 0;
    std::string reader, book;
    double rating, pred_rating, se = 0.0, baseline_se = 0.0;
    IDX_SCORE_VEC simi;
    auto test_data = input.test_data_vec();
    auto reader_to_idx = input.usr2idx();
    auto book_to_idx = input.item2idx();
    SP UIMAT;
    if(SP::IsRowMajor) {
//        std::sort(test_data.begin(), test_data.end(), [&reader_to_idx, &book_to_idx](const auto &a, const auto &b) {
//            if(reader_to_idx[std::get<0>(a)] == reader_to_idx[std::get<0>(b)])
//                return book_to_idx[std::get<1>(a)] < book_to_idx[std::get<1>(b)];
//            return reader_to_idx[std::get<0>(a)] < reader_to_idx[std::get<0>(b)];
//        });
        UIMAT = input.train_data_row();
    }
    else {
//        std::sort(test_data.begin(), test_data.end(), [&book_to_idx, &reader_to_idx](const auto &a, const auto &b) {
//            if(book_to_idx[std::get<1>(a)] == book_to_idx[std::get<1>(b)])
//                return reader_to_idx[std::get<0>(a)] < reader_to_idx[std::get<0>(b)];
//            return book_to_idx[std::get<1>(a)] < book_to_idx[std::get<1>(b)];
//        });
        UIMAT = input.train_data_col();
    }
//    std::cout << "test rmse sorting done" << std::endl;
    StopWatch sw; double elapsed;
    std::cout << "test time ... ";  sw.lap();
    for(auto & r : test_data) {
        std::tie(reader, book, rating) = r;
        if(SP::IsRowMajor) {
            idx = reader_to_idx.count(reader) ? reader_to_idx[reader] : -1;
            another_idx = book_to_idx.count(book) ? book_to_idx[book] : -1;
        } else {
            idx = book_to_idx.count(book) ? book_to_idx[book] : -1;
            another_idx = reader_to_idx.count(reader) ? reader_to_idx[reader] : -1;
        }

        if(idx == -1 || another_idx == -1) {
            pred_rating = avg_value;
            ++avg_count;
        } else {
            if(keep_nonzero_topk)
                simi = KNN<SP>::naive_kNearest(UIMAT, idx, another_idx, k, simi_th);
            else
                simi = KNN<SP>::naive_kNearest(UIMAT, idx, -1, k, simi_th);
            pred_rating = predict_ui_rating<SP>(another_idx, simi);
            if(pred_rating == std::numeric_limits<double>::max()) {
                pred_rating = avg_value;
                ++avg_count;
            }
        }
        se += pow(rating-pred_rating, 2);
        baseline_se += pow(rating-avg_value, 2);
    }
    double rmse = sqrt(se/(double)test_data.size());
    elapsed = sw.lap(); std::cout << elapsed << " sec" << std::endl;
    std::cout << "avg ratio : " << avg_count/(double)test_data.size() << std::endl;
    std::cout << "rmse : " << rmse << std::endl;
    std::cout << "baseline rmse : " << sqrt(baseline_se/(double)test_data.size()) << std::endl;

    return rmse;
}

ID_SCORE_VEC CF::recommend(const std::string & target, const std::string & id,
                            const std::string & based, int k, double simi_th, int n,
                            bool keep_nonzero_topk) {
    if(target == "user")
        return recommended_items_for_user(id, based, k, simi_th, n, keep_nonzero_topk);
    else
        return recommended_users_for_item(id, based, k, simi_th, n, keep_nonzero_topk);
}

template double CF::predict_ui_rating<SP_COL>(int idx, const IDX_SCORE_VEC &idx_score) const;
template double CF::predict_ui_rating<SP_ROW>(int idx, const IDX_SCORE_VEC &idx_score) const;

template double CF::test_rmse<SP_COL>(double avg_value, int k, double simi_th, bool keep_nonzero_topk);
template double CF::test_rmse<SP_ROW>(double avg_value, int k, double simi_th, bool keep_nonzero_topk);
