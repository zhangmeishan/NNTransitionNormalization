
include_directories(
  model
)

add_executable(GDSegmentor GDSegmentor.cpp)
target_link_libraries(GDSegmentor ${LIBS})
