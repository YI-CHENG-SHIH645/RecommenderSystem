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

double CF::predict_ui_rating(const std::string &ui, int idx, const IDX_SCORE_VEC & idx_score, bool ret_avg) {
    double ws = 0.0, k = 0.0;  // normalization
    SV vec;
    if(ui == "user") {
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
        if(ret_avg)
            return 5.5;
        return 0.0;
    };

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
            rating = predict_ui_rating("user", i, sorted_user_simi, false);
            if(rating != 0)
                weighted_sum.coeffRef(i) = rating;
        }
    }
    size_t num_non_zero = weighted_sum.nonZeros();
    IDX_SCORE_VEC itemID_score(num_non_zero);
    for(int i=0; i<num_non_zero; ++i) {
        itemID_score.emplace(itemID_score.begin(), *(weighted_sum.innerIndexPtr()+i), *(weighted_sum.valuePtr()+i));
    }
    std::sort(itemID_score.begin(), itemID_score.end(), comp_fn);
    itemID_score.resize(std::min(n, (int)itemID_score.size()));

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
            rating = predict_ui_rating("item", i, sorted_item_simi, false);
            if(rating != 0)
                weighted_sum.coeffRef(i) = rating;
        }
    }
    size_t num_non_zero = weighted_sum.nonZeros();
    IDX_SCORE_VEC userID_score(num_non_zero);
    for(int i=0; i<num_non_zero; ++i) {
        userID_score.emplace(userID_score.begin(), *(weighted_sum.innerIndexPtr()+i), *(weighted_sum.valuePtr()+i));
    }
    std::sort(userID_score.begin(), userID_score.end(), comp_fn);
    userID_score.resize(std::min(n, (int)userID_score.size()));

    elapsed = sw.lap(); std::cout << elapsed << " sec" << std::endl;

    std::cout << "# non-zero score recommendation items : " << num_non_zero << std::endl;
    std::cout << "top n recommendation items for the given user : result_n=" << num_non_zero << std::endl;
    for(auto item : userID_score)
        std::cout << item.first << " : " << item.second << std::endl;

    return userID_score;
}

//void CF::test_rmse(const std::string & method) {
//    long u_size = (long)((float)UIMAT_Row.rows() * test_u_size);
//    long i_size = (long)((float)UIMAT_Row.cols() * test_i_size);
//    SP_ROW test_Row_Block = UIMAT_Row.bottomRightCorner(u_size, i_size);
//    SP_COL test_Col_Block = UIMAT_Col.bottomRightCorner(u_size, i_size);
//    long u_start = UIMAT_Row.rows() - u_size;
//    long i_start = UIMAT_Row.cols() - i_size;
//
//    IDX_SCORE_VEC simi;  SV ws, test_Row_Block_row;
//    double squared_error = 0.0, rmse;
//    for(long u=u_start; u<UIMAT_Row.rows(); ++u) {
//        if(!UIMAT_Row.row(u).nonZeros()) continue;
//        std::cout << u << std::endl;
//        simi = calculate_simi("user", (int)u, 0.01, (int)i_start, (int)u_start);
//        ws = calculate_weighted_sum("user", simi);
//        test_Row_Block_row = test_Row_Block.row(u-u_start);
//        for(int i=0; i<test_Row_Block_row.nonZeros(); ++i) {
//            int idx = *(test_Row_Block_row.innerIndexPtr()+i);
//            std::cout << " " << idx << " " << *(test_Row_Block_row.valuePtr() + i) << " " << ws.coeff(idx + i_start) <<std::endl;
//            if(ws.coeff(idx + i_start) != 0) {
//                std::cout << "y_true : " << *(test_Row_Block_row.valuePtr() + i) << std::endl;
//                std::cout << "y_pred : " << ws.coeff(idx + i_start) << std::endl;
//                std::cout << "(a-b)2 : " << pow(*(test_Row_Block_row.valuePtr() + i) - ws.coeff(idx + i_start), 2) << std::endl;
//                squared_error += pow(*(test_Row_Block_row.valuePtr() + i) - ws.coeff(idx + i_start), 2);
//                std::cout << "(a-b)2 : " << squared_error << std::endl;
//                exit(0);
//            }
//        }
//    }
//    rmse = sqrt(squared_error/(double)test_Row_Block.nonZeros());
//    std::cout << "rmse : " << rmse << std::endl;
//}
