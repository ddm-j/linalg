add_executable(examples)

target_sources(examples
    PRIVATE
        examples.cpp
)

if(LINALG_BLOCKED_MATMUL)
    target_compile_definitions(examples PRIVATE BLOCKED_MATMUL_ENABLED)
endif()

target_link_libraries(examples
    PRIVATE
        linalg
)