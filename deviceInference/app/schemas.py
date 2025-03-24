from pydantic import BaseModel, Field

class GenericAPIResponse(BaseModel):
    """Generic model for the API response."""


class SuccessAPIResponse(GenericAPIResponse):
    """API reponse model for success message."""

    message: str


class ErrorAPIResponse(GenericAPIResponse):
    """Generic API response model for error message."""

    error_message: str

class FingerprintOutput(BaseModel):
    """Schema for fingerprint request."""

    sys_name: str = Field(description="Linux device name")
    kernel_version: str = Field(description="Kernel version of the Linux device")