#include <fstream>
#include <iostream>
#include <unordered_map>
#include <boost/algorithm/string/replace.hpp>
#include "CF.h"
#include "StopWatch.h"
#include "KNN.h"

auto comp_fn = [](const auto & a, const auto & b){ return a.second == b.second ? a.first < b.first : a.second > b.second;};

SP_COL get_UIMAT(const std::string & file_path) {

    std::ifstream fin(file_path);
    if(fin.fail()) {
        std::cout << "Error: " << strerror(errno) << " : " << file_path << std::endl;
        exit(0);
    }

    StopWatch sw;  double elapsed;

    std::cout << "read as vector of triplets ... ";  sw.lap();

    int num_reader = 0, num_book = 0;
    std::string reader, book, rating, line;

    std::unordered_map<std::string, int> book_to_idx;
    std::vector<Eigen::Triplet<double>> triplets;

    std::getline(fin, line); // skip header
    while(std::getline(fin, line)) {
        std::stringstream l(line);
        std::getline(l, reader, ';'); boost::replace_all(reader, "\"", "");
        std::getline(l, book, ';'); boost::replace_all(book, "\"", "");
        std::getline(l, rating, ';'); boost::replace_all(rating, "\"", "");
        num_reader = std::max(num_reader, std::stoi(reader));
        if(!book_to_idx.count(book))
            book_to_idx[book] = num_book++;
        triplets.emplace_back(Eigen::Triplet<double>(std::stoi(reader), book_to_idx[book], std::stoi(rating)+1));
    }

    elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;

    std::cout << "Init spmat (alloc & set) ... ";  sw.lap();

    SP_COL spmat_col(num_reader+1, num_book);
    spmat_col.setFromTriplets(triplets.begin(), triplets.end());

    elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;

    std::cout << "# readers (rows m) : " << num_reader << " , "
              << "# books (cols n) : " << num_book << std::endl;
    std::cout << "# non zero element: " << spmat_col.nonZeros() << '\n' << std::endl;

    return spmat_col;
}

void CF::filter_rare_scoring() {

    StopWatch sw;  double elapsed;

    std::cout << "filter out < 100 rating users ... ";  sw.lap();

    SV row;  int valid_user_count = 0;
    for(int i=0; i<UIMAT_Row.rows(); ++i) {
        row = UIMAT_Row.row(i);
        if (row.nonZeros() < 30) {
            for(int j=0; j<row.nonZeros(); ++j) {
                UIMAT_Row.coeffRef(i, *(row.innerIndexPtr()+j)) = 0;
            }
        } else {
            ++valid_user_count;
        }
    }
    UIMAT_Row.prune(1, 0);

    elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;
    std::cout << "After filtering users ... " << std::endl;
    std::cout << "# readers (rows m) : " << valid_user_count << std::endl;
    std::cout << "# non zero element: " << UIMAT_Row.nonZeros() << '\n' << std::endl;

    std::cout << "filter out < 100 rating items ... ";  sw.lap();

    SV col;  int valid_item_count = 0;
    for(int i=0; i<UIMAT_Col.cols(); ++i) {
        col = UIMAT_Col.col(i);
        if (col.nonZeros() < 30) {
            for(SP_COL::InnerIterator it(UIMAT_Col, i); it; ++it) {
                it.valueRef() = 0;
            }
        } else {
            ++valid_item_count;
        }
    }
    UIMAT_Col.prune(1, 0);

    elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;
    std::cout << "After filtering items ... " << std::endl;
    std::cout << "# items (rows m) : " << valid_item_count << std::endl;
    std::cout << "# non zero element: " << UIMAT_Col.nonZeros() << '\n' << std::endl;
}

template<typename SP>
double CF::predict_ui_rating(int idx, const IDX_SCORE_VEC & idx_score, bool ret_avg) {
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

    if(k==0) {
        if(ret_avg) {
            return 5.5;
        }
        return 0.0;
    }

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
        if(tgt_row.coeff(i) == 0) {
            rating = predict_ui_rating<SP_ROW>(i, sorted_user_simi, false);
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
        if(tgt_col.coeff(i) == 0) {
            rating = predict_ui_rating<SP_COL>(i, sorted_item_simi, false);
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
    if(block.IsRowMajor) {
        block = UIMAT_Row.bottomRightCorner(u_size, i_size);
        major_start = UIMAT_Row.rows() - u_size;
        major_bound = UIMAT_Row.rows();
        minor_start = UIMAT_Row.cols() - i_size;
    } else {
        block = UIMAT_Col.bottomRightCorner(u_size, i_size);
        major_start = UIMAT_Col.cols() - i_size;
        major_bound = UIMAT_Col.cols();
        minor_start = UIMAT_Col.rows() - u_size;
    }
    std::cout << block.nonZeros() << std::endl;
    SV block_vec;  double se = 0.0, rating;
    IDX_SCORE_VEC simi;
    for(long i=major_start; i<major_bound; ++i) {
        if(block.IsRowMajor) {
            block_vec = block.row(i - major_start);
            simi = KNN<SP>::naive_kNearest(UIMAT_Row, i, k, simi_th, minor_start);
        }
        else {
            block_vec = block.col(i - major_start);
            simi = KNN<SP>::naive_kNearest(UIMAT_Col, i, k, simi_th, minor_start);
        }

        for(long j=0; j<block_vec.nonZeros(); ++j) {
            long idx = *(block_vec.innerIndexPtr()+j);
            double val = *(block_vec.valuePtr()+j);
            rating = predict_ui_rating<SP>(idx+minor_start, simi, true);
            se += pow(rating - val, 2);
        }
    }
    std::cout << se << " / " << (double)block.nonZeros() << std::endl;
    double rmse = sqrt(se/(double)block.nonZeros());

    return rmse;
}

template double CF::predict_ui_rating<SP_COL>(int idx, const IDX_SCORE_VEC &idx_score, bool ret_avg);
template double CF::predict_ui_rating<SP_ROW>(int idx, const IDX_SCORE_VEC &idx_score, bool ret_avg);

template double CF::calculate_rmse<SP_COL>(int k, double simi_th, double test_u_size, double test_i_size);
template double CF::calculate_rmse<SP_ROW>(int k, double simi_th, double test_u_size, double test_i_size);
