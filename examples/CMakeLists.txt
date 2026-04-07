add_executable(sandbox)

target_sources(sandbox
    PRIVATE
        sandbox.cpp
)

target_link_libraries(sandbox
    PRIVATE
        linalg
)