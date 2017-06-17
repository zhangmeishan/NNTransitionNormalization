
include_directories(
  model
)

add_executable(NNSegmentor NNSegmentor.cpp)
target_link_libraries(NNSegmentor ${LIBS})
