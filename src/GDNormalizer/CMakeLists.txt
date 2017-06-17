
include_directories(
  model
)

add_executable(GDNormalizer GDNormalizer.cpp)
target_link_libraries(GDNormalizer ${LIBS})

