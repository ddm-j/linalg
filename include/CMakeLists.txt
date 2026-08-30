add_library(linalg INTERFACE)

target_sources(linalg
    INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
        linalg/Matrix.h
        linalg/stride_view.h
        linalg/errors.h
        linalg/decomposition.h
        linalg/solvers.h
)

target_include_directories(linalg
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_features(linalg INTERFACE cxx_std_23)

if(LINALG_BLOCKED_MATMUL)
    target_compile_definitions(linalg INTERFACE BLOCKED_MATMUL_ENABLED)
endif()