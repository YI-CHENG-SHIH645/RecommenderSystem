#include <iostream>
#include "KNN.h"

auto comp_fn = [](const auto & a, const auto & b){ return a.second == b.second ? a.first < b.first : a.second > b.second;};

template<typename SP>
IDX_SCORE_VEC KNN<SP>::calculate_simi(const SP & sp_mat, int idx, double simi_th, int vec_test_start, int mat_test_start) {
    // return vec of non-zero simi -> idx : simi
    IDX_SCORE_VEC simi;
    SV vec, tgt;

    if(sp_mat.IsRowMajor) {
        tgt = sp_mat.row(idx);
    } else {
        tgt = sp_mat.col(idx);
    }

    if(vec_test_start > -1) {
        for(int i=vec_test_start; i<tgt.size(); ++i) {
            if(tgt.coeffRef(i) != 0)
                tgt.coeffRef(i) = 0;
        }
    }

    double sqrt_bright = sqrt(tgt.cwiseProduct(tgt).sum());
    for(int j=0; j<sp_mat.outerSize(); ++j) {
        if(j==idx) continue;
        if(vec_test_start>-1 && j>=mat_test_start) break;
        if(sp_mat.IsRowMajor) vec = sp_mat.row(j);
        else                  vec = sp_mat.col(j);
        double top = tgt.cwiseProduct(vec).sum();
        if(top != 0) {
            double sqrt_bleft = sqrt(vec.cwiseProduct(vec).sum());
            double s = top / (sqrt_bleft * sqrt_bright);
            if(s >= simi_th)
                simi.emplace_back(j, s);
        }
    }

    return simi;
}

template<typename SP>
IDX_SCORE_VEC KNN<SP>::naive_kNearest(const SP & sp_mat, int idx, int k, double simi_th) {

    IDX_SCORE_VEC simi = calculate_simi(sp_mat, idx, simi_th);
    std::sort(simi.begin(), simi.end(), comp_fn);
    simi.resize(std::min(k, (int)simi.size()));

    return simi;
}

template class KNN<SP_COL>;
template class KNN<SP_ROW>;
