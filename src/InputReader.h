#pragma once
#include <StopWatch.h>
#include <iostream>
#include <fstream>
#include <KNN.h>
#include <unordered_map>
#include <boost/algorithm/string/replace.hpp>
#include <set>
#include <utility>

typedef std::unordered_map<std::string, int> NAME2IDX;
typedef std::unordered_map<int, std::string> IDX2NAME;
typedef std::vector<std::tuple<std::string, std::string, double>> VEC_TUPLE;

class InputReader {

private:
    std::string train_file_name, test_file_name;
    NAME2IDX user_to_idx, item_to_idx;
    IDX2NAME idx_to_user, idx_to_item;
    SP_COL train_mat_col;
    SP_ROW train_mat_row;
    VEC_TUPLE test_data;
    bool filter = false;
    std::set<long> valid_rows, valid_cols;

    std::ifstream open_train_file() {
        std::ifstream fin;
        fin.open(train_file_name);
        if(fin.fail()) {
            std::cout << train_file_name << " file could not be opened" << std::endl;
            exit(0);
        }
        return fin;
    }
    std::ifstream open_test_file() {
        std::ifstream fin;
        fin.open(test_file_name);
        if(fin.fail()) {
            std::cout << test_file_name << " file could not be opened" << std::endl;
            exit(0);
        }
        return fin;
    }
public:
    explicit InputReader(std::string file_name, std::string test_file_name)
    : train_file_name(std::move(file_name)), test_file_name(std::move(test_file_name)) {}

    explicit InputReader(const SP_COL & mat): train_mat_col(mat), train_mat_row(SP_ROW(mat)){
        for(int i=0; i<mat.cols(); ++i) {
            item_to_idx[std::to_string(i)] = i;
            idx_to_item[i] = std::to_string(i);
        }
        for(int i=0; i<mat.rows(); ++i) {
            user_to_idx[std::to_string(i)] = i;
            idx_to_user[i] = std::to_string(i);
        }
    }

    void parse(const std::string & mode, const char* dlm, bool skip_header, bool plus1) {
        std::ifstream fin;
        if(mode == "train") open_train_file().swap(fin); else open_test_file().swap(fin);
        int num_user = 0, num_item = 0;
        std::string user, item, rating, line;
        double numeric_rating;
        std::vector<Eigen::Triplet<double>> triplets;
        if(skip_header)
            std::getline(fin, line); // skip header
        StopWatch sw;  double elapsed;
        std::cout << "read as a vector of triplets (" << mode << ") ... ";  sw.lap();
        while(std::getline(fin, line)) {
            std::stringstream l(line);
            std::getline(l, user, *dlm); boost::replace_all(user, "\"", "");
            std::getline(l, item, *dlm); boost::replace_all(item, "\"", "");
            std::getline(l, rating, *dlm); boost::replace_all(rating, "\"", "");
            numeric_rating = plus1 ? std::stod(rating)+1 : std::stod(rating);
            if(mode == "train") {
                if (!user_to_idx.count(user)) {
                    user_to_idx[user] = num_user;
                    idx_to_user[num_user++] = user;
                }
                if (!item_to_idx.count(item)) {
                    item_to_idx[item] = num_item;
                    idx_to_item[num_item++] = item;
                }
                triplets.emplace_back(Eigen::Triplet<double>(user_to_idx[user], item_to_idx[item], numeric_rating));
            }
            else
                test_data.emplace_back(std::make_tuple(user, item, numeric_rating));
        }
        elapsed = sw.lap();  std::cout << elapsed << " sec" << std::endl;

        if(mode == "train") {
            std::cout << "Init SP matrix (alloc & set) ... ";  sw.lap();
            train_mat_col = SP_COL(num_user, num_item);
            train_mat_col.setFromTriplets(triplets.begin(), triplets.end());
            train_mat_row = SP_ROW(train_mat_col);
            elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;
            std::cout << "# readers (rows m) : " << num_user << " , "
                      << "# books (cols n) : " << num_item << std::endl;
            std::cout << "# non zero element: " << train_mat_col.nonZeros() << std::endl;
            std::cout << "density = " << (double)train_mat_col.nonZeros()/(num_user*num_item) << '\n' << std::endl;
        }
        else {
            std::cout << "length of test data : " << test_data.size() << std::endl;
        }
    }

    void filter_user(int th) {
        filter = true;
        StopWatch sw;  double elapsed;
        std::cout << "filter out rarely rated user ... ";  sw.lap();

        for(long i=0; i<train_mat_row.rows(); ++i) {
            if (train_mat_row.row(i).nonZeros() > th) {
                valid_rows.insert(i);
            }
        }
        elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;
        std::cout << "After filtering users ... " << std::endl;
        std::cout << "# readers (rows m) : " << train_mat_row.rows() << " -> " << valid_rows.size() << std::endl;
    }

    void filter_item(int th) {
        filter = true;
        StopWatch sw;  double elapsed;
        std::cout << "filter out rarely rated item ... ";  sw.lap();

        for(long i=0; i<train_mat_col.cols(); ++i) {
            if (train_mat_col.col(i).nonZeros() > th) {
                valid_cols.insert(i);
            }
        }
        elapsed = sw.lap();  std::cout << elapsed << " sec\n" << std::endl;
        std::cout << "After filtering items ... " << std::endl;
        std::cout << "# items (cols m) : " << train_mat_col.cols() << " -> " << valid_cols.size() << std::endl;
    }

    NAME2IDX & usr2idx() { return user_to_idx; }
    NAME2IDX & item2idx() { return item_to_idx; }
    IDX2NAME & idx2usr() { return idx_to_user; }
    IDX2NAME & idx2item() { return idx_to_item; }
    const SP_COL & train_data_col() { return train_mat_col; }
    const SP_ROW & train_data_row() { return train_mat_row; }
    const VEC_TUPLE & test_data_vec() { return test_data; }
    const std::set<long> & valid_col_idx() { return valid_cols; }
    const std::set<long> & valid_row_idx() { return valid_rows; }
    [[nodiscard]] bool filtered() const { return filter; }
};
