# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from kvikio._lib import cufile_driver  # type: ignore

# TODO: Wrap nicely, maybe as a dataclass?
DriverProperties = cufile_driver.DriverProperties


def driver_open() -> None:
    """Open the cuFile driver

    cuFile accept multiple calls to `driver_open`, only the first call
    opens the driver, but every call should have a matching call to
    `driver_close`.

    Raises
    ------
    RuntimeError
        If cuFile isn't available.
    """
    return cufile_driver.driver_open()


def driver_close() -> None:
    """Close the cuFile driver

    cuFile accept multiple calls to `driver_open`, only the first call
    opens the driver, but every call should have a matching call to
    driver_close().

    Raises
    ------
    RuntimeError
        If cuFile isn't available.
    """
    return cufile_driver.driver_close()
