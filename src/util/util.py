# Utility class to help code for python

# Import libs
import glob as _glob


# Debug condition.
_SHOW_DEBUG_FOR_READ_FILE = False
_SHOW_DEBUG = True


class Util:
    # Read file from @filePath and return the list of string for each lines.
    @staticmethod
    def readFile(filePath):
        lines = []
        if filePath == None:
            return lines
        file = open(filePath)
        for line in file:
            lines.append(line.strip())
        file.close()
        if _SHOW_DEBUG_FOR_READ_FILE:
            print('Reading file: ' + filePath)
            for each in lines:
                print(each)
        return lines

    # Read all docs in the file.
    # @folderPath is the string file path e. '*.txt'.
    # @numFiles to read. If it's 5 then it will read 5 files.
    # If it is less then or equal to 0, read all files.
    # returns the list of docs with list of lines in the file.
    @staticmethod
    def readAllFilesInFolder(folderPath, numFiles=0):
        docs = []
        if folderPath == None:
            return docs
        countFiles = 0
        for files in _glob.glob(folderPath):
            # Check if it need to read number of files.
            if numFiles > 0:
                countFiles += 1
            # Read the files
            file = Util.readFile(files)
            lines = []
            for line in file:
                lines.append(line.strip())
            docs.append(lines)
            # Check the number of files to read.
            if countFiles >= numFiles:
                break
        # Print for debuggin.
        if _SHOW_DEBUG:
            fileNum = 0
            for doc in docs:
                fileNum += 1
                print('Doc#: ' + str(fileNum))
                for line in docs:
                    print('line: ' + str(line))
                print()
        return docs
