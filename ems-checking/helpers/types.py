# Format types response
def Ok(message = "Success", data = [], version = "1.0.0"):
    '''Define format of success message'''
    return {
        "message": message, 
        "data": data, 
        "version": version
    }

def Error(message = "Error", error = [], version = "1.0.0"):
    '''Define format of error message'''
    return {
        "message": message, 
        "errors": error, 
        "version": version
    }