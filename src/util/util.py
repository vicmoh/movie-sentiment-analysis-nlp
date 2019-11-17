# Author: Vicky Mohammad
# Description: File containing helper function
# other utilities.

# Import libs
import glob as _glob


# Debug condition.
_SHOW_DEBUG_FOR_READ_FILE = False
_SHOW_DEBUG_FOR_READ_ALL_FILE = True


class Util:
    @staticmethod
    def readFile(filePath):
        """Read file from @filePath and return the list of string for each lines."""
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

    @staticmethod
    def readAllFilesInFolder(
        folderPath,
        numFiles=0,
        eachLineCallback=None
    ):
        """Read all docs in the file.
        @folderPath is the string file path e. '*.txt'.
        @numFiles to read. If it's 5 then it will read 5 files.
        @eachLineCallBack param callbacks the line it's currently parsing
        If it is less then or equal to 0, read all files.
        returns the list of docs with list of lines in the file."""
        # Check if path exist and setup
        docs = []
        if folderPath == None:
            return docs
        countFiles = 1
        fileNames = _glob.glob(folderPath)

        # Loop through the files
        for fileName in fileNames:
            if _SHOW_DEBUG_FOR_READ_ALL_FILE:
                print(fileName)
            # Check if it need to read number of files.
            if numFiles > 0:
                countFiles += 1

            # Read the files
            file = Util.readFile(fileName)
            lines = []
            for line in file:
                if eachLineCallback is not None:
                    line = eachLineCallback(line)
                lines.append(line.strip())
            docs.append(lines)
            # Check the number of files to read.
            if countFiles >= numFiles:
                break

        # Print for debuggin.
        if _SHOW_DEBUG_FOR_READ_ALL_FILE:
            print()
            fileNum = 0
            for doc in docs:
                fileNum += 1
                print('Doc#: ' + str(fileNum))
                print(doc)
                print()
        return docs
