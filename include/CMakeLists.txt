add_library(linalg INTERFACE)

target_sources(linalg
    INTERFACE
    FILE_SET HEADERS
    FILES
        linalg/Matrix.h
        linalg/stride_view.h
        linalg/decomposition.h
        linalg/solvers.h
)

target_compile_features(linalg INTERFACE cxx_std_23)