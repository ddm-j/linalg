add_executable(examples)

target_sources(examples
    PRIVATE
        examples.cpp
)

target_link_libraries(examples
    PRIVATE
        linalg
)