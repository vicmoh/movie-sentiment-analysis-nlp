# Utility class to help code for python
# Import libs
import glob


# Debug condition.
_SHOW_DEBUG = True


class Util:
    def __init__(self):
        super().__init__()

    # Read file from @filePath and return the list of string for each lines.
    def readFile(self, filePath):
        lines = []
        if filePath == None:
            return lines
        file = open(filePath)
        for line in file:
            lines.append(line.strip())
        file.close()
        if _SHOW_DEBUG:
            print('Reading file: ' + filePath)
            for each in lines:
                print(each)
        return lines

    # Read all docs in the file.
    # @folderPath is the string file path e. '*.txt'.
    # @numFiles to read. If it's 5 then 5it will read 5 files.
    # If it is -1
    # returns the list of docs with list of lines in the file.
    def readAllFilesInFolder(self, folderPath, numFiles=-1):
        docs = []
        if folderPath == None:
            return docs
        numFiles = 0
        for files in glob.glob(folderPath):
            numFiles += 1
            file = self.readFile(files)
            lines = []
            for line in file:
                lines.append(line.strip())
            docs.append(lines)
            # Check the number of files to read.
            if numFiles < 0:
                continue
            elif numFiles >= numFiles:
                break
        # Print for debuggin.
        if _SHOW_DEBUG:
            fileNum = 0
            for doc in docs:
                fileNum += 1
                print('Doc#: ' + str(fileNum))
                for line in docs:
                    print('line: ' + str(line))
        return docs
