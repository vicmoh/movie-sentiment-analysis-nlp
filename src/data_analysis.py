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
    listOfAvgSenPerDoc = []
    docNames = []

    def __init__(self, folderPath=None):
        """This object is used to process the data analysis.
        If @folderPath is empty. It does nothing."""
        super().__init__()
        if folderPath is None:
            return
        self.__readAllFilesInFolder(folderPath=folderPath)

    def printCombinedResultWith(self, obj):
        """Print a combined data of 2 data. It is used to
        combined pos and neg value data together."""
        # Init data
        totalDocs = self.totalDocs + obj.totalDocs
        totalSen = self.totalSentence + obj.totalSentence
        totalToks = self.totalToken + obj.totalToken
        minSen, maxSen, minTok, maxTok = 0, 0, 0, 0
        # Determine min sentence
        if self.minSentence < obj.minSentence:
            minSen = self.minSentence
        else:
            minSen = obj.minSentence
        # Determine max the sentence
        if self.maxSentence > obj.maxSentence:
            maxSen = self.maxSentence
        else:
            maxSen = obj.maxSentence
        # Determine min the token
        if self.minToken < obj.minToken:
            minTok = self.minToken
        else:
            minTok = obj.minToken
        # Determine max token
        if self.maxToken > obj.maxToken:
            maxTok = self.maxToken
        else:
            maxTok = obj.maxToken
        # Print the combined data
        print('Number of documents: ', totalDocs)
        print('Min sentences: ', minSen)
        print('Max sentences: ', maxSen)
        print('Avg sentences: ', totalSen / totalDocs)
        print('Total Sentences: ', totalSen)
        print('Min tokens: ', minTok)
        print('Max tokens: ', maxTok)
        print('Avg tokens: ', totalToks / totalDocs)
        print('Total tokens: ', totalToks)

    def printResult(self):
        """Print the data analysis result."""
        print('Number of documents: ', self.totalDocs)
        print('Min sentences: ', self.minSentence)
        print('Max sentences: ', self.maxSentence)
        print('Avg sentences: ', self.totalSentence / self.totalDocs)
        print('Total Sentences: ', self.totalSentence)
        print('Min tokens: ', self.minToken)
        print('Max tokens: ', self.maxToken)
        print('Avg tokens: ', self.totalToken / self.totalDocs)
        print('Total tokens: ', self.totalToken)

    def __resetData(self):
        """Reset all the attributes to the default values."""
        self.totalDocs = 0
        self.minSentence = 0
        self.maxSentence = 0
        self.totalSentence = 0
        self.minToken = 0
        self.maxToken = 0
        self.totalToken = 0

    def __readFile(self, filePath):
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

    def __trackTokenData(self, num):
        """Track and count token data for the analysis."""
        self.totalToken += num
        if num != 0 and num > self.maxToken:
            self.maxToken = num
        if num != 0 and self.minToken == 0:
            self.minToken = num
        elif num != 0 and num < self.minToken:
            self.minToken = num

    def __trackSentencesData(self, num):
        """Track and count token data for the analysis."""
        self.totalSentence += num
        if num != 0 and num > self.maxSentence:
            self.maxSentence = num
        if num != 0 and self.minSentence == 0:
            self.minSentence = num
        elif num != 0 and num < self.minSentence:
            self.minSentence = num

    def __readAllFilesInFolder(
        self,
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
        self.totalDocs += len(fileNames)
        # Loop through the files
        for fileName in fileNames:
            if _SHOW_DEBUG_FOR_READ_ALL_FILE:
                print(fileName)
            # Read the files
            file = self.__readFile(fileName)
            lines = []
            countLines = 0
            for line in file:
                countLines += 1
                if eachLineCallback is not None:
                    line = eachLineCallback(line)
                lines.append(line.strip())
                # keep track of number of lines
            self.__trackSentencesData(countLines)
            docs.append(lines)
        # Track the number sentence in preprocess data
        preData = SkLearn.preprocess(docs)
        count = 0
        for each in preData:
            tokens = len(re.split('[ \r\n]+|[ \n]+', each))
            lines = len(re.split('[\r\n]+|[\n]+', each))
            self.__trackTokenData(tokens)
            self.listOfAvgSenPerDoc.append({
                'fileName': fileNames[count],
                'avg':  (tokens / lines),
                'tokens': tokens,
                'lines': lines
            })
            count += 1
        return docs
