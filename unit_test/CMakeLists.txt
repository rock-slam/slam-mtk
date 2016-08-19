include_directories( ${EIGEN3_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR})

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/slom)
include_directories( ${CXSPARSE_INCLUDE_DIR} )


MACRO(MAKE_TEST name scalar)
    ADD_TEST(${name}_${scalar} ${name}_${scalar})
    ADD_EXECUTABLE(${name}_${scalar} ${name}.cc)
    SET_TARGET_PROPERTIES(${name}_${scalar} PROPERTIES COMPILE_FLAGS -DSCALAR=${scalar})
    TARGET_LINK_LIBRARIES(${name}_${scalar} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY} )
ENDMACRO()


MAKE_TEST(mtk_test float)
MAKE_TEST(mtk_test double)
MAKE_TEST(slom_choleskytest double)
MAKE_TEST(ukfom_test double)

