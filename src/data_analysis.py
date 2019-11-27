# Author: Vicky Mohammad
# Description: File containing helper function
# other utilities.

# Import libs
import glob as _glob
from sk_learn import SkLearn
from nltk.stem import WordNetLemmatizer
import re


# Debug condition.
_SHOW_DEBUG_FOR_READ_FILE = False
_SHOW_DEBUG_FOR_READ_ALL_FILE = False


class DataAnalysis:
    totalDocs = 0
    # For tracking sentences
    minSentence = 0
    maxSentence = 0
    totalSentence = 0
    # For tracking tokens
    minToken = 0
    maxToken = 0
    totalToken = 0

    @staticmethod
    def resetData():
        DataAnalysis.totalDocs = 0
        DataAnalysis.minSentence = 0
        DataAnalysis.maxSentence = 0
        DataAnalysis.totalSentence = 0
        DataAnalysis.minToken = 0
        DataAnalysis.maxToken = 0
        DataAnalysis.totalToken = 0

    @staticmethod
    def printResult():
        print('\nCalculating data analysis...')
        print('Number of documents: ', DataAnalysis.totalDocs)
        print('Min sentences: ', DataAnalysis.minSentence)
        print('Max sentences: ', DataAnalysis.maxSentence)
        print('Avg sentences: ', DataAnalysis.totalSentence / DataAnalysis.totalDocs)
        print('Total Sentences: ', DataAnalysis.totalSentence)
        print('Min tokens: ', DataAnalysis.minToken)
        print('Max tokens: ', DataAnalysis.maxToken)
        print('Avg tokens: ', DataAnalysis.totalToken / DataAnalysis.totalDocs)
        print('Total tokens: ', DataAnalysis.totalToken)

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
        # Debug printing
        if _SHOW_DEBUG_FOR_READ_FILE:
            print('Reading file: ' + filePath)
            for each in lines:
                print(each)
        return lines

    @staticmethod
    def trackTokenData(num):
        DataAnalysis.totalToken += num
        if num != 0 and num > DataAnalysis.maxToken:
            DataAnalysis.maxToken = num
        if num != 0 and DataAnalysis.minToken == 0:
            DataAnalysis.minToken = num
        elif num != 0 and num < DataAnalysis.minToken:
            DataAnalysis.minToken = num

    @staticmethod
    def trackSentencesData(num):
        DataAnalysis.totalSentence += num
        if num != 0 and num > DataAnalysis.maxSentence:
            DataAnalysis.maxSentence = num
        if num != 0 and DataAnalysis.minSentence == 0:
            DataAnalysis.minSentence = num
        elif num != 0 and num < DataAnalysis.minSentence:
            DataAnalysis.minSentence = num

    @staticmethod
    def readAllFilesInFolder(
        folderPath,
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
        countFiles = 0
        fileNames = _glob.glob(folderPath)
        DataAnalysis.totalDocs += len(fileNames)

        # Loop through the files
        for fileName in fileNames:
            if _SHOW_DEBUG_FOR_READ_ALL_FILE:
                print(fileName)
            # Read the files
            file = DataAnalysis.readFile(fileName)
            lines = []
            countLines = 0
            for line in file:
                countLines += 1
                if eachLineCallback is not None:
                    line = eachLineCallback(line)
                lines.append(line.strip())
                # keep track of number of lines
            DataAnalysis.trackSentencesData(countLines)
            docs.append(lines)

        # Track the number sentence in preprocess data
        preData = SkLearn.preprocess(docs)
        for each in preData:
            tokens = re.split('[ \r\n]+|[ \n]+', each)
            DataAnalysis.trackTokenData(len(tokens))
        return docs
