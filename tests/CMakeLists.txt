set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

find_package(GTest REQUIRED)
set(link_lib GTest::gtest)

aux_source_directory(../tests DIR_TEST)

add_executable(unittest ${DIR_TEST})
target_include_directories(unittest PUBLIC ${GTest_INCLUDE_DIR})
target_link_libraries(unittest ${link_lib} xllm)