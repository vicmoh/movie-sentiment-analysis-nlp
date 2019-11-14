
_SHOW_DEBUG = True
# Utility class to help code for python


class Util:
    def __init__(self):
        super().__init__()
    # Read file from @filePath and return the list of string for each lines
    def readFile(filePath):
        lines = []
        if filePath == None:
            return lines
        file = open(filePath)
        for line in file:
            lines.append(line)
        file.close()
        if _SHOW_DEBUG:
            print("Reading file: " + filePath)
            for each in lines:
                print(each)
        return lines
