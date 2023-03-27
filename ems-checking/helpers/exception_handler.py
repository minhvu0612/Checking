# Handle exception 
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi import Request
import json
# Package
from .types import Ok, Error


# Exception type Define
class ExceptionType:
    '''Define exception type in project'''
    def __init__(self, http_code, message, data = [], error = [], version = "1.0.0"):
        self.http_code = http_code
        self.message = message
        self.data = data
        self.error = error
        self.version = version

    def getHttpCode(self):
        return self.http_code

    def getMessage(self):
        return self.message

    def getData(self):
        return self.data

    def getError(self):
        return self.error

    def getResponse(self):
        if self.http_code == 200:
            return Ok(message = self.message, data = self.data, version = self.version)
        else:
            return Error(message = self.message, error = self.error, version = self.version)


# Overide http exception
class EMSException(HTTPException):
    '''Overide HTTPException'''
    pass


# Define dafault response
async def not_found(request: Request, exc: EMSException):
    return JSONResponse(status_code = 404, content = Error(message = "NOT_FOUND"))

async def internal_server_error(request: Request, exc: EMSException):
    return JSONResponse(status_code = 500, content = Error(message = "INTERNAL_SERVER_ERROR"))


# Exception default
exception_types = {
    404: not_found,
    500: internal_server_error
}