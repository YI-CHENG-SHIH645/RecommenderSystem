#include <fstream>
#include <iostream>
#include <unordered_map>
#include <boost/algorithm/string/replace.hpp>
#include "CF.h"
#include "StopWatch.h"
#include "KNN.h"

auto comp_fn = [](const auto & a, const auto & b){
    return a.second == b.second ? a.first < b.first : a.second > b.second;
};

SP_COL CF::get_UIMAT(const std::string & file_path, const std::string & test_file_path,
                     const char* dlm, bool skip_header) {

    // open train file
    std::ifstream fin(file_path);
    if(fin.fail()) {
        std::cout << "Error: " << strerror(errno) << " : " << file_path << std::endl;
        exit(0);
    }
    // open test file if provided
    std::ifstream test_fin;
    if(!test_file_path.empty()) {
        test_fin.open(test_file_path);
        if(test_fin.fail()) {
            std::cout << "Error: " << strerror(errno) << " : " << file_path << std::endl;
            exit(0);
        }
    }

    StopWatch sw;  double elapsed;

    std::cout << "read as a vector of triplets ... ";  sw.lap();
    int num_reader = 0, num_book = 0;
    std::string reader, book, rating, line;
    std::vector<Eigen::Triplet<double>> triplets;

    if(skip_header)
        std::getline(fin, line); // skip header
    while(std::getline(fin, line)) {
        std::stringstream l(line);
        std::getline(l, reader, *dlm); boost::replace_all(reader, "\"", "");
        std::getline(l, book, *dlm); boost::replace_all(book, "\"", "");
        std::getline(l, rating, *dlm); boost::replace_all(rating, "\"", "");
        num_reader = std::max(num_reader, std::stoi(reader));
        if(!book_to_idx.count(book))
            book_to_idx[book] = num_book++;
        triplets.emplace_back(Eigen::Triplet<double>(std::stoi(reader), book_to_idx[book], std::stoi(rating)));
    }

    if(!test_file_path.empty()) {
        if(skip_header)
            std::getline(test_fin, line); // skip header
        while(std::getline(test_fin, line)) {
            std::stringstream l(line);
            std::getline(l, reader, '\t'); boost::replace_all(reader, "\"", "");
            std::getline(l, book, '\t'); boost::replace_all(book, "\"", "");
            std::getline(l, rating, '\t'); boost::replace_all(rating, "\"", "");
            test_data.emplace_back(std::make_tuple(std::stoi(reader), book, std::stoi(rating)));
        }
    }

    elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;

    std::cout << "Init spmat (alloc & set) ... ";  sw.lap();
    SP_COL spmat_col(num_reader+1, num_book);
    spmat_col.setFromTriplets(triplets.begin(), triplets.end());

    elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;

    std::cout << "# readers (rows m) : " << num_reader << " , "
              << "# books (cols n) : " << num_book << std::endl;
    std::cout << "# non zero element: " << spmat_col.nonZeros() << '\n' << std::endl;
    std::cout << "density = " << (double)spmat_col.nonZeros()/(num_reader*num_book) << '\n' << std::endl;

    return spmat_col;
}

void CF::filter_rare_scoring(int user_th, int item_th) {

    StopWatch sw;  double elapsed;

    std::cout << "filter out rarely rated users ... ";  sw.lap();
    SV row;
    // instead of set the elements of row to zero, just mark the row
    for(int i=0; i<UIMAT_Row.rows(); ++i) {
        row = UIMAT_Row.row(i);
        if (row.nonZeros() > user_th) {
            valid_rows.insert(i);
        }
    }
    elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;

    std::cout << "After filtering users ... " << std::endl;
    std::cout << "# readers (rows m) : " << UIMAT_Row.rows() << " -> " << valid_rows.size() << std::endl;

    std::cout << "filter out rarely rated items ... ";  sw.lap();
    SV col;
    // instead of set the elements of col to zero, just mark the col
    for(int i=0; i<UIMAT_Col.cols(); ++i) {
        col = UIMAT_Col.col(i);
        if (col.nonZeros() > item_th) {
            valid_cols.insert(i);
        }
    }
    elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;

    std::cout << "After filtering items ... " << std::endl;
    std::cout << "# items (cols m) : " << UIMAT_Col.cols() << " -> " << valid_cols.size() << std::endl;
}

template<typename SP>
double CF::predict_ui_rating(int idx, const IDX_SCORE_VEC & idx_score) {
    SV vec;
    double ws = 0.0, k = 0.0;  // normalization
    if(SP::IsRowMajor) {
        vec = UIMAT_Col.col(idx);
    } else {
        vec = UIMAT_Row.row(idx);
    }

    for(auto idx_simi : idx_score) {
        if(vec.coeff(idx_simi.first) == 0) continue;
        ws += vec.coeff(idx_simi.first) * idx_simi.second;
        k += abs(idx_simi.second);
    }

    if(k==0)
        return std::numeric_limits<double>::min();

    return ws/k;
}

IDX_SCORE_VEC CF::recommended_items_for_user(int user_id, int k, double simi_th, int n) {

    StopWatch sw; double elapsed;

    std::cout << "calculate weighted sum of top k item vectors ... ";  sw.lap();
    IDX_SCORE_VEC sorted_user_simi = KNN<SP_ROW>::naive_kNearest(UIMAT_Row, user_id, k, simi_th);
//    SV weighted_sum = calculate_weighted_sum("user", sorted_user_simi);
    SV weighted_sum(UIMAT_Row.cols());
    SV tgt_row = UIMAT_Row.row(user_id);
    double rating;
    for(int i=0; i<tgt_row.size(); ++i) {
        if(tgt_row.coeff(i) == 0 && (valid_cols.count(i) || !filter)) {
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
    for(auto item : itemID_score)
        std::cout << item.first << " : " << item.second << std::endl;

    return itemID_score;
}

IDX_SCORE_VEC CF::recommended_users_for_item(int item_id, int k, double simi_th, int n) {

    StopWatch sw; double elapsed;

    std::cout << "calculate weighted sum of top k user vectors ... ";  sw.lap();
    IDX_SCORE_VEC sorted_item_simi = KNN<SP_COL>::naive_kNearest(UIMAT_Col, item_id, k, simi_th);
//    SV weighted_sum = calculate_weighted_sum("item", sorted_item_simi);
    SV weighted_sum(UIMAT_Col.rows());
    SV tgt_col = UIMAT_Col.col(item_id);
    double rating;
    for(int i=0; i<tgt_col.size(); ++i) {
        if(tgt_col.coeff(i) == 0 && (valid_rows.count(i) || !filter)) {
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
    for(auto item : userID_score)
        std::cout << item.first << " : " << item.second << std::endl;

    return userID_score;
}

template<typename SP>
double CF::calculate_rmse(int k, double simi_th, double test_u_size, double test_i_size) {
    long u_size = (long)((double)UIMAT_Col.rows() * test_u_size);
    long i_size = (long)((double)UIMAT_Col.cols() * test_i_size);
    SP block;
    long major_start, major_bound, minor_start;
    std::set<long> rc;
    if(block.IsRowMajor) {
        block = UIMAT_Row.bottomRightCorner(u_size, i_size);
        major_start = UIMAT_Row.rows() - u_size;
        major_bound = UIMAT_Row.rows();
        minor_start = UIMAT_Row.cols() - i_size;
        rc = valid_cols;
    } else {
        block = UIMAT_Col.bottomRightCorner(u_size, i_size);
        major_start = UIMAT_Col.cols() - i_size;
        major_bound = UIMAT_Col.cols();
        minor_start = UIMAT_Col.rows() - u_size;
        rc = valid_rows;
    }
    std::cout << "total non-zero : " << block.nonZeros() << std::endl;
//    std::cout << "total non-zero : " << *(valid_cols.rbegin()) << std::endl;

    SV block_vec;
    long idx; double val, rating;
    double se = 0.0, baseline_se = 0.0;
    int rate_avg_count = 0, valid_counts = 0;
    bool has_user_simi = false;
    IDX_SCORE_VEC simi;

    for(long i=major_start; i<major_bound; ++i) {
        std::cout << i << " / " << major_bound << std::endl;
        if(block.IsRowMajor)
            block_vec = block.row(i - major_start);
        else
            block_vec = block.col(i - major_start);
        // 累計 block 內符合資格的 ui element 數量
        for(long j=0; j<block_vec.nonZeros(); ++j) {
            idx = *(block_vec.innerIndexPtr()+j);
            val = *(block_vec.valuePtr()+j);
            if(rc.count((int)(idx+minor_start)) || !filter) {
                if(!has_user_simi) {
                    if(SP::IsRowMajor)
                        simi = KNN<SP>::naive_kNearest(UIMAT_Row, i, k, simi_th, minor_start);
                    else
                        simi = KNN<SP>::naive_kNearest(UIMAT_Col, i, k, simi_th, minor_start);
                    has_user_simi = true;
                }
                ++valid_counts;
                rating = predict_ui_rating<SP>(idx + minor_start, simi);
                if(rating == std::numeric_limits<double>::min())
                    rating = 5.5;
                se += pow(rating - val, 2);
                baseline_se += pow(5.5 - val, 2);
                if (rating == 5.5)
                    ++rate_avg_count;
            }
        }
        has_user_simi = false;
    }

    std::cout << se << " / " << valid_counts << std::endl;
    double rmse =          sqrt(se/valid_counts);
    double baseline_rmse = sqrt(baseline_se/valid_counts);
    std::cout << "baseline rmse ... " << baseline_rmse << std::endl;
    std::cout << "rate of avg   ... " << rate_avg_count/valid_counts << std::endl;

    return rmse;
}

template<typename SP>
double CF::test_rmse(int k, double simi_th) {
    int reader, idx, another_idx, cur_idx=-1;
    std::string book;
    double rating, pred_rating, se = 0.0, baseline_se = 0.0;
    IDX_SCORE_VEC simi;

    if(SP::IsRowMajor) {
        std::sort(test_data.begin(), test_data.end(), [](const auto &a, const auto &b) {
            return std::get<0>(a) < std::get<0>(b);
        });
    }
    else {
        std::sort(test_data.begin(), test_data.end(), [this](const auto &a, const auto &b) {
            return book_to_idx[std::get<1>(a)] < book_to_idx[std::get<1>(b)];
        });
    }

    for(auto & r : test_data) {
        std::tie(reader, book, rating) = r;
        idx = SP::IsRowMajor ? reader : book_to_idx.count(book) ? book_to_idx[book] : -1;
        another_idx = !SP::IsRowMajor ? reader : book_to_idx.count(book) ? book_to_idx[book] : -1;
        if(idx == -1)
            pred_rating = 3;
        else if(idx != cur_idx) {
            cur_idx = idx;
            simi = SP::IsRowMajor ? KNN<SP>::naive_kNearest(UIMAT_Row, idx, k, simi_th) :
                    KNN<SP>::naive_kNearest(UIMAT_Col, idx, k, simi_th);
            pred_rating = predict_ui_rating<SP>(another_idx, simi);
        } else {
            pred_rating = predict_ui_rating<SP>(another_idx, simi);
        }
        se += pow(rating-pred_rating, 2);
        baseline_se += pow(3-rating, 2);
    }
    double rmse = sqrt(se/(double)test_data.size());
    std::cout << "rmse : " << rmse << std::endl;
    std::cout << "baseline rmse : " << sqrt(baseline_se/(double)test_data.size()) << std::endl;

    return rmse;
}


template double CF::predict_ui_rating<SP_COL>(int idx, const IDX_SCORE_VEC &idx_score);
template double CF::predict_ui_rating<SP_ROW>(int idx, const IDX_SCORE_VEC &idx_score);

template double CF::calculate_rmse<SP_COL>(int k, double simi_th, double test_u_size, double test_i_size);
template double CF::calculate_rmse<SP_ROW>(int k, double simi_th, double test_u_size, double test_i_size);

template double CF::test_rmse<SP_COL>(int k, double simi_th);
template double CF::test_rmse<SP_ROW>(int k, double simi_th);
