# This needs some clean-up before publication, sorry
# ADD_EXECUTABLE( dlr_data_set dlr/dlr_data_set.cpp)
# TARGET_LINK_LIBRARIES( dlr_data_set ${Boost_PROGRAM_OPTIONS_LIBRARY})
ADD_EXECUTABLE(sba sba/slom_sba.cc)
target_link_libraries(sba PUBLIC slom Eigen3::Eigen)
target_include_directories(sba PRIVATE
  "$<TARGET_PROPERTY:mtk,INTERFACE_INCLUDE_DIRECTORIES>"
  ${CXSPARSE_INCLUDE_DIR}
)

ADD_EXECUTABLE(relation relation/relation.cpp)
target_link_libraries(relation PUBLIC slom Boost::boost Boost::program_options Eigen3::Eigen)
target_include_directories(relation PRIVATE
  "$<TARGET_PROPERTY:mtk,INTERFACE_INCLUDE_DIRECTORIES>"
  ${CXSPARSE_INCLUDE_DIR}
)

ADD_EXECUTABLE(relation_slameval relation/relation_slameval.cpp)
target_link_libraries(relation_slameval PUBLIC slom)
target_include_directories(relation_slameval PRIVATE
  "$<TARGET_PROPERTY:mtk,INTERFACE_INCLUDE_DIRECTORIES>"
  ${CXSPARSE_INCLUDE_DIR}
)
