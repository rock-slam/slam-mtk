cmake_minimum_required(VERSION 2.6)
project(mtk)

add_definitions(-Wall -Wno-unused-function -march=native -msse2 -mssse3 -mfpmath=sse)
option(SLOM_BUILD_EXAMPLES "Build the example code" ON)
if (SLOM_BUILD_LIBONLY)
    set(SLOM_BUILD_EXAMPLES OFF)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)

FIND_PACKAGE(Boost 1.36.0 REQUIRED)
FIND_PACKAGE(CXSparse REQUIRED) 
find_package(PkgConfig)
pkg_check_modules(Eigen3 REQUIRED eigen3)

INCLUDE_DIRECTORIES(${CXSPARSE_INCLUDE_DIR} )
INCLUDE_DIRECTORIES(${Eigen3_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} )

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

ADD_LIBRARY(slom slom/src/Estimator.cpp slom/src/SparseFunction.cpp)


TARGET_LINK_LIBRARIES(slom
	${CXSPARSE_LIBRARY}
)

IF(SLOM_BUILD_EXAMPLES)
    FIND_PACKAGE(Boost 1.36.0 REQUIRED COMPONENTS program_options unit_test_framework)
    ENABLE_TESTING()
    ADD_EXECUTABLE( min_example doc/example.cpp)
    TARGET_LINK_LIBRARIES(min_example slom)

    ADD_SUBDIRECTORY (example)
    ADD_SUBDIRECTORY (unit_test)
ENDIF()

INSTALL(FILES
    mtk/startIdx.hpp
    mtk/build_manifold.hpp
    mtk/mean_and_covar.hpp
    DESTINATION include/mtk)

INSTALL(FILES
    mtk/src/eigen.hpp
    mtk/src/mtkmath.hpp
    mtk/src/SubManifold.hpp
    mtk/src/vectview.hpp
    DESTINATION include/mtk/src)

INSTALL(FILES
    mtk/types/pose.hpp
    mtk/types/S2.hpp
    mtk/types/SOn.hpp
    mtk/types/vect.hpp
    DESTINATION include/mtk/types)

INSTALL(FILES
    ukfom/mtkwrap.hpp
    ukfom/util.hpp
    ukfom/ukf.hpp
    DESTINATION include/ukfom)

INSTALL(FILES
    ukfom/traits/dof.hpp
    DESTINATION include/ukfom/traits)

install(TARGETS slom
    DESTINATION lib)

SET(PKG_CONFIG_EXPORT_LIBS "-lslom")
configure_file(mtk.pc.in mtk.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/mtk.pc DESTINATION lib/pkgconfig)
