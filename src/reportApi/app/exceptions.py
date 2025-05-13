class ReportApiError(Exception):
    """Base class for API exception."""


class ServiceFunctionError(ReportApiError):
    """Base class for error occured in service functions."""


class InvalidKernelError(ServiceFunctionError):
    """Given kernel version is not in the database."""


class InvalidDeviceNameError(ServiceFunctionError):
    """Given device system name is not in the database."""


class InvalidModelIdError(ServiceFunctionError):
    """Given model id is not in the database."""


class InvalidDatasetIdError(ServiceFunctionError):
    """Given dataset id is not in the database."""


class DatabaseError(ReportApiError):
    """Base class for error occured during communication with database."""
