#pragma once
#include <vector>

// SP : type of eigen sparse matrix
template<typename SP>
class CF {
private:
    SP UIMAT;
//  SP IIMAT;  void cal_IIMAT();
    SP get_UIMAT(const std::string & file_path);
    std::vector<std::pair<int, double>> naive_kNearest_user(int user, int k, double simi_th);
    std::vector<std::pair<int, double>> naive_kNearest_item(int item, int k, double simi_th);
public:
    explicit CF(const std::string & file_path): UIMAT(get_UIMAT(file_path)) {};
    std::vector<std::pair<int, double>> recommendation_for_a_user(int user_id, int k=30, double simi_th=0.01);
    std::vector<std::pair<int, double>> an_item_to_users(int item, int k=30, double simi_th=0.01);
};
