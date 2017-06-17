
include_directories(
  model
)

add_executable(NNNormalizer NNNormalizer.cpp)
target_link_libraries(NNNormalizer ${LIBS})

