cmake_minimum_required(VERSION 3.9)
project(recommender_system)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp)
add_compile_options(-D NDEBUG)
add_executable(${CMAKE_PROJECT_NAME}_run ${SOURCE_FILES})
find_package(pybind11 REQUIRED)
pybind11_add_module(recommender_system create_so.cpp)
set_target_properties(recommender_system PROPERTIES LIBRARY_OUTPUT_DIRECTORY ..)

include_directories(src)
include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include)
include_directories(/opt/homebrew/Cellar/boost/1.76.0/include)

add_subdirectory(src)
add_subdirectory(tests)

target_link_libraries(${CMAKE_PROJECT_NAME}_run ${CMAKE_PROJECT_NAME}_lib)
target_link_libraries(recommender_system PRIVATE ${CMAKE_PROJECT_NAME}_lib)

# Doxygen Build
option(BUILD_DOC "Build Documentation" ON)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(BUILD_DOC_DIR ${CMAKE_SOURCE_DIR}/build/docs)
    if(NOT EXISTS ${BUILD_DOC_DIR})
        file(MAKE_DIRECTORY ${BUILD_DOC_DIR})
    endif()

    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    message("Doxygen build started")
    add_custom_target(Doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
else(DOXYGEN_FOUND)
    message("Doxygen needs to be installed to generate the documentation.")
endif(DOXYGEN_FOUND)