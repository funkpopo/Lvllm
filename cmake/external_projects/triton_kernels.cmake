# Install OpenAI triton_kernels from https://github.com/triton-lang/triton/tree/main/python/triton_kernels

set(DEFAULT_TRITON_KERNELS_TAG "v3.6.0")

# Set TRITON_KERNELS_SRC_DIR for use with local development with vLLM. We expect TRITON_KERNELS_SRC_DIR to
# be directly set to the triton_kernels python directory.
if (DEFINED ENV{TRITON_KERNELS_SRC_DIR})
  message(STATUS "[triton_kernels] Fetch from $ENV{TRITON_KERNELS_SRC_DIR}")
  FetchContent_Declare(
          triton_kernels
          SOURCE_DIR $ENV{TRITON_KERNELS_SRC_DIR}
  )

else()
  set(TRITON_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/third_party/triton")
  if(NOT EXISTS "${TRITON_SUBMODULE_DIR}/python/triton_kernels/triton_kernels")
    message(FATAL_ERROR
      "[triton_kernels] Triton source not found at ${TRITON_SUBMODULE_DIR}. "
      "Please initialize submodules (git submodule update --init --recursive) "
      "or set TRITON_KERNELS_SRC_DIR.")
  endif()
  find_package(Git QUIET)
  if(GIT_FOUND AND EXISTS "${TRITON_SUBMODULE_DIR}/.git")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${TRITON_SUBMODULE_DIR}" describe --tags --exact-match
      RESULT_VARIABLE TRITON_TAG_RESULT
      OUTPUT_VARIABLE TRITON_TAG
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT TRITON_TAG_RESULT EQUAL 0 OR NOT TRITON_TAG STREQUAL "${DEFAULT_TRITON_KERNELS_TAG}")
      message(FATAL_ERROR
        "[triton_kernels] Triton must be pinned to tag ${DEFAULT_TRITON_KERNELS_TAG}, "
        "but current checkout is '${TRITON_TAG}'.")
    endif()
  endif()
  message (STATUS "[triton_kernels] Use submodule at ${TRITON_SUBMODULE_DIR}:${DEFAULT_TRITON_KERNELS_TAG}")
  FetchContent_Declare(
          triton_kernels
          # TODO (varun) : Fetch just the triton_kernels directory from Triton
          SOURCE_DIR ${TRITON_SUBMODULE_DIR}
          SOURCE_SUBDIR python/triton_kernels/triton_kernels
  )
endif()

# Fetch content
FetchContent_MakeAvailable(triton_kernels)

if (NOT triton_kernels_SOURCE_DIR)
  message (FATAL_ERROR "[triton_kernels] Cannot resolve triton_kernels_SOURCE_DIR")
endif()

if (DEFINED ENV{TRITON_KERNELS_SRC_DIR})
  set(TRITON_KERNELS_PYTHON_DIR "${triton_kernels_SOURCE_DIR}/")
else()
  set(TRITON_KERNELS_PYTHON_DIR "${triton_kernels_SOURCE_DIR}/python/triton_kernels/triton_kernels/")
endif()

message (STATUS "[triton_kernels] triton_kernels is available at ${TRITON_KERNELS_PYTHON_DIR}")

add_custom_target(triton_kernels)

# Ensure the vllm/third_party directory exists before installation
install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/third_party/triton_kernels\")")

## Copy .py files to install directory.
install(DIRECTORY
        ${TRITON_KERNELS_PYTHON_DIR}
        DESTINATION
        vllm/third_party/triton_kernels/
        COMPONENT triton_kernels
        FILES_MATCHING PATTERN "*.py")
