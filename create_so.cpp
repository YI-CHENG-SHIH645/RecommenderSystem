#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <InputReader.h>
#include <CF.h>

namespace py = pybind11;

PYBIND11_MODULE(recommender_system, m) {
    py::class_<InputReader>(m, "InputReader")
            .def(py::init<std::string, std::string>())
            .def("parse", &InputReader::parse)
            .def("filter_user", &InputReader::filter_user)
            .def("filter_item", &InputReader::filter_item);
    py::class_<CF>(m, "CF")
            .def(py::init<InputReader &>())
            .def("recommend", &CF::recommend,
                 py::arg("target"), py::arg("id"), py::arg("based"),
                 py::arg("k")=-1, py::arg("simi_th")=0, py::arg("n")=10,
                 py::arg("keep_nonzero_topk")=true)
            .def("user_based_rmse", &CF::test_rmse<SP_ROW>)
            .def("item_based_rmse", &CF::test_rmse<SP_COL>);
}
