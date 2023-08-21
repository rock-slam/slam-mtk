cmake_minimum_required(VERSION 3.5)
project(mtk VERSION 1.0.0)

add_definitions(-Wall -Wno-unused-function -march=native -msse2 -mssse3 -mfpmath=sse)
option(SLOM_BUILD_EXAMPLES "Build the example code" ON)

if(SLOM_BUILD_LIBONLY)
    set(SLOM_BUILD_EXAMPLES OFF)
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)

find_package(Boost 1.36.0 REQUIRED)
find_package(CXSparse REQUIRED)
find_package(PkgConfig)
find_package(Eigen3 REQUIRED)

add_library(mtk INTERFACE)
add_library(mtk::mtk ALIAS mtk)

target_include_directories(
    mtk INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${CXSPARSE_INCLUDE_DIR}
)
target_link_libraries(mtk INTERFACE Boost::boost Eigen3::Eigen)

add_library(slom STATIC src/slom/Estimator.cpp src/slom/SparseFunction.cpp)

target_include_directories(
    slom PRIVATE
    "$<TARGET_PROPERTY:mtk,INTERFACE_INCLUDE_DIRECTORIES>"
    ${CXSPARSE_INCLUDE_DIR}
)

TARGET_LINK_LIBRARIES(slom
    Eigen3::Eigen
    ${CXSPARSE_LIBRARY}
)

IF(SLOM_BUILD_EXAMPLES)
    FIND_PACKAGE(Boost 1.36.0 REQUIRED COMPONENTS program_options unit_test_framework)
    ENABLE_TESTING()
    ADD_EXECUTABLE(min_example doc/example.cpp)
    TARGET_LINK_LIBRARIES(min_example mtk slom)

    ADD_SUBDIRECTORY(example)

    # ADD_SUBDIRECTORY (unit_test)
ENDIF()

set(${PROJECT_NAME}_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE STRING "Path to install ${PROJECT_NAME} Config*.cmake files to.")
set(${PROJECT_NAME}_MODULE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake" CACHE STRING "Path to install ${PROJECT_NAME}'s .cmake  module files to.")

write_basic_package_version_file(
    ${PROJECT_NAME}ConfigVersion.cmake # The name of the version file needed by find_package.
    VERSION ${PROJECT_VERSION} # The version of the project, already set by the `project` command at the top of this file
    COMPATIBILITY SameMajorVersion # We use semantic versioning, backward compatibity is only guaranteed for a same major version
)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in # This is your template file
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake # This is the resulting file
    INSTALL_DESTINATION ${${PROJECT_NAME}_INSTALL_CMAKEDIR} # This is where the file will be installed

    # List of paths that needs to be relocated once installed
    # For example if the variable containing the path is named MY_PATH, all instances of @PACKAGE_MY_PATH@ in the template will be replaced by the relocated version of the path
    # This is mostly only needed when you want to install cmake modules or have an unusual layout that cmake is not aware of.
    PATH_VARS ${PROJECT_NAME}_MODULE_INSTALL_DIR # This will be exposed as @PACKAGE_BoilerPlate_MODULE_INSTALL_DIR@ in the template file

    # Imported targets do not require the following macros
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
install(DIRECTORY include/${PROJECT_NAME} include/ukfom include/slom
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(
    TARGETS mtk slom
    EXPORT ${PROJECT_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
    EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${${PROJECT_NAME}_INSTALL_CMAKEDIR})

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION
    ${${PROJECT_NAME}_INSTALL_CMAKEDIR}
)

SET(PKG_CONFIG_EXPORT_LIBS "-lslom ${CXSPARSE_LIBRARY}")
SET(PKG_CONFIG_INCLUDE_DIRS "-I${CXSPARSE_INCLUDE_DIR}")
configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/mtk.pc DESTINATION lib/pkgconfig)
