# Install/export packaging for ITLabAI

set(_itlabai_vendor_inc_dir "${CMAKE_INSTALL_INCLUDEDIR}/itlabai/thirdparty")
set(_itlabai_vendor_lib_dir "${CMAKE_INSTALL_LIBDIR}/itlabai/thirdparty/lib")

# Install public headers
install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
)

# Export in-tree targets
install(TARGETS
  itlabai_features
  itlabai_openmp
  itlabai_graph_lib
  itlabai_graph_transformations_lib
  itlabai_layers_lib
  itlabai_layers_onednn_lib
  itlabai_perf_lib
  itlabai_reader_lib
  EXPORT ITLabAITargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

# Bundle vendored third-party headers
install(DIRECTORY "${ONEDNN_INSTALL_DIR}/include/" DESTINATION "${_itlabai_vendor_inc_dir}/onednn")
install(DIRECTORY "${TBB_INSTALL_DIR}/include/" DESTINATION "${_itlabai_vendor_inc_dir}/tbb")
install(DIRECTORY "${OPENCV_INSTALL_DIR}/include/" DESTINATION "${_itlabai_vendor_inc_dir}/opencv")
if(ITLABAI_ENABLE_KOKKOS)
  install(DIRECTORY "${KOKKOS_INSTALL_DIR}/include/" DESTINATION "${_itlabai_vendor_inc_dir}/kokkos")
endif()
install(DIRECTORY "${PROJECT_SOURCE_DIR}/3rdparty/Json/include/" DESTINATION "${_itlabai_vendor_inc_dir}/json")

# Bundle vendored third-party libraries.
#
# Note: On ELF platforms, installing only $<TARGET_FILE:...> can copy a symlink
# without the versioned "real" .so file, which breaks consumers at link time.
# Prefer copying the relevant library family from the external install tree.
if(WIN32)
  # Runtime + import library.
  install(FILES "$<TARGET_FILE:dnnl>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
  install(FILES "$<TARGET_LINKER_FILE:dnnl>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)

  install(FILES "$<TARGET_FILE:TBB::tbb>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
  install(FILES "$<TARGET_LINKER_FILE:TBB::tbb>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)

  install(FILES "$<TARGET_FILE:OpenCV::opencv_world>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
  install(FILES "$<TARGET_LINKER_FILE:OpenCV::opencv_world>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
else()
  # Copy the whole versioned chain (e.g. libtbb.so, libtbb.so.12, libtbb.so.12.13).
  install(DIRECTORY "${ONEDNN_INSTALL_DIR}/lib/" DESTINATION "${_itlabai_vendor_lib_dir}"
    FILES_MATCHING
    PATTERN "libdnnl.so*"
    PATTERN "libdnnl.dylib*"
    PATTERN "cmake" EXCLUDE
    PATTERN "pkgconfig" EXCLUDE
  )
  install(DIRECTORY "${TBB_INSTALL_DIR}/lib/" DESTINATION "${_itlabai_vendor_lib_dir}"
    FILES_MATCHING
    PATTERN "libtbb*.so*"
    PATTERN "libtbb*.dylib*"
    PATTERN "cmake" EXCLUDE
    PATTERN "pkgconfig" EXCLUDE
  )
  install(DIRECTORY "${OPENCV_INSTALL_DIR}/lib/" DESTINATION "${_itlabai_vendor_lib_dir}"
    FILES_MATCHING
    PATTERN "libopencv_world.so*"
    PATTERN "libopencv_world.dylib*"
    PATTERN "cmake" EXCLUDE
    PATTERN "pkgconfig" EXCLUDE
  )
endif()

if(ITLABAI_ENABLE_KOKKOS)
  install(FILES "$<TARGET_FILE:kokkoscore_external>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
  install(FILES "$<TARGET_FILE:kokkoscontainers_external>" DESTINATION "${_itlabai_vendor_lib_dir}" OPTIONAL)
endif()

# Package config
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ITLabAIConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/ITLabAIConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ITLabAIConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ITLabAI"
)

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/ITLabAIThirdPartyTargets.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ITLabAIThirdPartyTargets.cmake"
  @ONLY
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ITLabAIConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ITLabAIConfigVersion.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ITLabAIThirdPartyTargets.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ITLabAI"
)

install(EXPORT ITLabAITargets
  NAMESPACE ITLabAI::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ITLabAI"
)
