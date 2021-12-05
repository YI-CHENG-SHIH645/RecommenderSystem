#include <iostream>
#include <fstream>
#include <unordered_map>
#include <StopWatch.h>
#include <CollaborativeFiltering.h>
#include <eigen3/Eigen/Sparse>
#include <boost/algorithm/string/replace.hpp>

template<typename SP>
SP CF<SP>::get_UIMAT(const std::string & file_path) {
    std::ifstream fin(file_path);
    if(fin.fail()) {
        std::cout << "Error: " << strerror(errno) << " : " << file_path << std::endl;
        exit(0);
    }

    int num_reader = 0, num_book = 0;
    std::string reader, book, rating, line;
    std::unordered_map<std::string, int> book_to_idx; // book name -> index
    std::vector<Eigen::Triplet<double>> triplets;

    StopWatch sw;  double elapsed;

    std::cout << "read as vector of triplets ... ";
    sw.lap();
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
    elapsed = sw.lap();
    std::cout << elapsed << " sec\n" << std::endl;

    std::cout << "Init spmat (alloc & set) ... ";
    sw.lap();
    SP spmat(num_reader, num_book);
    spmat.setFromTriplets(triplets.begin(), triplets.end());
    elapsed = sw.lap();
    std::cout << elapsed << " sec" << std::endl;

    std::cout << "# readers (rows m) : " << num_reader << " , "
              << "# books (cols n) : " << num_book << std::endl;
    std::cout << "# non zero element: " << spmat.nonZeros() << '\n' << std::endl;

    return spmat;
}

//template<typename SP>
//void CF<SP>::cal_IIMAT() {
//
//    StopWatch sw; double elapsed;
//
//
//    std::cout << "transpose ... ";
//    sw.lap();
//    SP IUMAT = UIMAT.transpose();
//    elapsed = sw.lap();
//    std::cout << elapsed << " sec" << std::endl;
//
//
//    std::cout << "cal top ... ";
//    sw.lap();
//    SP top = (IUMAT * UIMAT).template selfadjointView<Eigen::Upper>();
//    elapsed = sw.lap();
//    std::cout << elapsed << " sec" << std::endl;
//    std::cout << "# non zero element: " << top.nonZeros() << '\n' << std::endl;
//
//
//    std::cout << "cal bottom ... ";
//    sw.lap();
//    std::map<std::pair<int, int>, std::pair<double, double>> item_pair_bottom_lr;
//    for(int k=0; k<UIMAT.outerSize(); ++k) {
//        for(typename SP::InnerIterator it1(UIMAT, k); it1; ++it1) {
//            for(typename SP::InnerIterator it2=it1; it2; ++it2) {
//                std::pair<int, int> item_pair(it1.col(), it2.col());
//                item_pair_bottom_lr[item_pair].first += pow(it1.value(), 2);
//                item_pair_bottom_lr[item_pair].second += pow(it2.value(), 2);
//            }
//        }
//    }
//    SP bottom(UIMAT.cols(), UIMAT.cols()), bottom_symmetric;
//    std::vector<Eigen::Triplet<double>> triplets;
//    for(auto const & p: item_pair_bottom_lr) {
//        double l_sq_sum = p.second.first;
//        double r_sq_sum = p.second.second;
//        double bottom_element = sqrt(l_sq_sum * r_sq_sum);
//        triplets.emplace_back(Eigen::Triplet<double>(p.first.first, p.first.second, bottom_element));
//    }
//    bottom.setFromTriplets(triplets.begin(), triplets.end());
//    bottom_symmetric = bottom.template selfadjointView<Eigen::Upper>();
//    elapsed = sw.lap();
//    std::cout << elapsed << " sec" << std::endl;
//    std::cout << "# non zero element: " << bottom_symmetric.nonZeros() << '\n' << std::endl;
//    std::string check_infinite = std::isfinite(bottom_symmetric.sum()) ? " ... no -> OK " : " yes -> something went wrong ";
//    std::cout << "check inf value exist : " << check_infinite << std::endl;
//
//
//    std::cout << "elementwise div ... ";
//    sw.lap();
//    IIMAT = top.cwiseQuotient(bottom_symmetric);
//    elapsed = sw.lap();
//    std::cout << elapsed << " sec" << std::endl;
//    std::cout << "# non zero element: " << IIMAT.nonZeros() << '\n' << std::endl;
//    check_infinite = std::isfinite(IIMAT.sum()) ? " ... no -> OK " : " yes -> something went wrong ";
//    std::cout << "check inf value exist : " << check_infinite << std::endl;
//    std::string check_nonZeros = (top.nonZeros() == bottom_symmetric.nonZeros())
//            && (bottom_symmetric.nonZeros() == IIMAT.nonZeros()) ? " ... all eq -> OK " : " exist not eq -> something went wrong ";
//    std::cout << "check # non-zero eq : " << check_nonZeros << std::endl;
//}

template<typename SP>
std::vector<std::pair<int, double>> CF<SP>::naive_kNearest_user(int user, int k, double simi_th) {

    StopWatch sw; double elapsed;

    sw.lap();
    // check index
    Eigen::SparseMatrix<double, Eigen::RowMajor> UIMAT_rowMajor(UIMAT);
    Eigen::SparseVector<double> tgt_row = UIMAT_rowMajor.row(user);
    Eigen::SparseVector<double> row;
    std::vector<std::pair<int, double>> simi;
    double sqrt_bright = sqrt(tgt_row.cwiseProduct(tgt_row).sum());

    for(int j=0; j<UIMAT_rowMajor.outerSize(); ++j) {
        if(j==user) continue;
        row = UIMAT_rowMajor.row(j);
        double top = tgt_row.cwiseProduct(row).sum();
        if(top != 0) {
            double sqrt_bleft = row.cwiseProduct(row).sum();
            double s = top / (sqrt_bleft * sqrt_bright);
            if(s >= simi_th)
                simi.emplace_back(std::make_pair(j, s));
        }
    }

    elapsed = sw.lap();
//    std::cout << elapsed << " sec" << std::endl;
//    std::cout << simi.size() << std::endl;
    std::sort(simi.begin(), simi.end(), [](const auto & a, const auto & b){
        return a.second == b.second ? a.first < b.first : a.second > b.second;
    });
    if(simi.size() > k)
        simi.resize(k);

//    for(auto pair : simi) {
//        std::cout << pair.first << "  " << pair.second << std::endl;
//    }
//    std::cout << simi.size() << std::endl;

    return simi;
}

template<typename SP>
std::vector<std::pair<int, double>> CF<SP>::naive_kNearest_item(int item, int k, double simi_th) {

    StopWatch sw; double elapsed;

    sw.lap();
    // check index
    Eigen::SparseVector<double> tgt_col = UIMAT.col(item);
    Eigen::SparseVector<double> col;
    std::vector<std::pair<int, double>> simi;
    double sqrt_bright = sqrt(tgt_col.cwiseProduct(tgt_col).sum());

    for(int j=0; j<UIMAT.innerSize(); ++j) {
        if(j==item) continue;

        col = UIMAT.col(j);
        double top = tgt_col.cwiseProduct(col).sum();
        if(top != 0) {
            double sqrt_bleft = col.cwiseProduct(col).sum();
            double s = top / (sqrt_bleft * sqrt_bright);
            if(s >= simi_th)
                simi.emplace_back(std::make_pair(j, s));
        }
    }

    elapsed = sw.lap();
//    std::cout << elapsed << " sec" << std::endl;
//    std::cout << simi.size() << std::endl;
    std::sort(simi.begin(), simi.end(), [](const auto & a, const auto & b){
        return a.second == b.second ? a.first < b.first : a.second > b.second;
    });
    if(simi.size() > k)
        simi.resize(k);

//    for(auto pair : simi) {
//        std::cout << pair.first << "  " << pair.second << std::endl;
//    }
//    std::cout << simi.size() << std::endl;

//    for(typename SP::InnerIterator it(UIMAT, 174747); it; ++it)
//        std::cout << it.value() << "(" << it.index() << ")  ";
//    std::cout << std::endl;
//    for(typename SP::InnerIterator it(UIMAT, 237075); it; ++it)
//        std::cout << it.value() << "(" << it.index() << ")  ";
//    std::cout << std::endl;
    return simi;
}

template<typename SP>
std::vector<std::pair<int, double>> CF<SP>::recommendation_for_a_user(int user, int k, double simi_th) {
    std::vector<std::pair<int, double>> sorted_simi_users = naive_kNearest_user(user, k, simi_th);
    Eigen::SparseVector<double> row, weighted_sum;
    for(auto idx_simi : sorted_simi_users) {
        row = UIMAT.row(idx_simi.first);
        weighted_sum += idx_simi.second * row;
    }
    std::cout << "# non-zero score recommendation items : " << weighted_sum.nonZeros() << std::endl;
    // 照推薦分數排序推薦
    std::vector<std::pair<int, double>> itemID_score(weighted_sum.nonZeros());
    for(int n=0; n<weighted_sum.nonZeros(); ++n) {
        itemID_score.emplace(itemID_score.begin(), *(weighted_sum.innerIndexPtr()+n), *(weighted_sum.valuePtr()+n));
    }
    itemID_score.resize(weighted_sum.nonZeros());
    std::sort(itemID_score.begin(), itemID_score.end(), [](const auto & a, const auto & b){
        return a.second == b.second ? a.first < b.first : a.second > b.second;
    });

    return itemID_score;
}

template<typename SP>
std::vector<std::pair<int, double>> CF<SP>::an_item_to_users(int item, int k, double simi_th) {
    std::vector<std::pair<int, double>> sorted_simi_items = naive_kNearest_item(item, k, simi_th);
    Eigen::SparseVector<double> col, weighted_sum;
    for(auto idx_simi : sorted_simi_items) {
        col = UIMAT.col(idx_simi.first);
        weighted_sum += idx_simi.second * col;
    }
    std::cout << "# non-zero score recommendation users : " << weighted_sum.nonZeros() << std::endl;
    // 照推薦分數排序推薦
    std::vector<std::pair<int, double>> userID_score(weighted_sum.nonZeros());
    for(int n=0; n<weighted_sum.nonZeros(); ++n) {
        userID_score.emplace(userID_score.begin(), *(weighted_sum.innerIndexPtr() + n), *(weighted_sum.valuePtr() + n));
    }
    userID_score.resize(weighted_sum.nonZeros());
    std::sort(userID_score.begin(), userID_score.end(), [](const auto & a, const auto & b){
        return a.second == b.second ? a.first < b.first : a.second > b.second;
    });

    return userID_score;
}

template class CF<typename Eigen::SparseMatrix<double>>;
