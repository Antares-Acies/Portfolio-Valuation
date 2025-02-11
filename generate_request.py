class RequestData:
    def __init__(self, path, path_info, host, user, scheme, tz, is_anonymous):
        self.path = path
        self.path_info = path_info
        self.host = host
        self.user = user
        self.scheme = scheme
        self.tz = tz
        self.is_anonymous = is_anonymous
 
    def update_data(self, path=None, path_info=None, host=None, user=None, scheme=None, tz=None, is_anonymous=None):
        if path is not None:
            self.path = path
        if path_info is not None:
            self.path_info = path_info
        if host is not None:
            self.host = host
        if user is not None:
            self.user = user
        if scheme is not None:
            self.scheme = scheme
        if tz is not None:
            self.tz = tz
        if is_anonymous is not None:
            self.is_anonymous = is_anonymous
 
    def __str__(self):
        return f"path: {self.path}, path_info: {self.path_info}, host: {self.host}, user: {self.user}, scheme: {self.scheme}, tz: {self.tz}, is_anonymous: {self.is_anonymous}"
 
# Example usage
user = "anil"  # Replace with the actual user object
request_data = RequestData('/users/App3/Build/computationModule/', '/users/App3/Build/computationModule/', '127.0.0.1:8080', user, 'http', 'Asia/Calcutta', True)
 
# # Print the data
# print(request_data)
 
# # Update the data
# request_data.update_data(path='/new/path/', host='192.168.1.1:8080')
 
# # Print the updated data
# print(request_data)