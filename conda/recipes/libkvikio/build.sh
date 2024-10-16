# Copyright (c) 2022, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory
mkdir cpp/build
pushd cpp/build
cmake .. \
      -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
      ${CMAKE_EXTRA_ARGS}
make
make install
popd
