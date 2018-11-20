using System;
using System.Collections.Generic;
using System.IO;

namespace hw3
{
    class MainClass
    {
        public static string startWord = "*START*";
        public static string stopWord = "*STOP*";
        public static int numIterations = 20;

        public static Random rand = new Random();


        public static void Main(string[] args)
        {

            if (args.Length == 1)
            {
                var start = DateTime.Now;

                var trainDataPath = @"../../conll03_ner/eng.train.small";
                var trainDataResult = @"../../conll03_ner/eng.train.small.result";

                var devDataPath = @"../../conll03_ner/eng.dev.small";
                var devDataResult = @"../../conll03_ner/eng.dev.small.result";

                var testDataPath = @"../../conll03_ner/eng.test.small";
                var testDataResult = @"../../conll03_ner/eng.test.small.result";

                getFeatureClassesFromFile(trainDataPath, out HashSet<string> posSet, out HashSet<string> wordSet, out HashSet<string> chunkSet, out HashSet<string> nerTagSet);


                var initialWeights = initializeWeights(nerTagSet, posSet, wordSet, chunkSet);

                var trainedWeights = train(trainDataPath, numIterations, initialWeights, nerTagSet);

                switch (args[0])
                {
                    case "train":
                        evaluate(trainDataPath, trainDataResult, trainedWeights, nerTagSet);
                        break;
                    case "dev":
                        evaluate(devDataPath, devDataResult, trainedWeights, nerTagSet);
                        break;
                    case "test":
                        evaluate(testDataPath, testDataResult, trainedWeights, nerTagSet);
                        break;

                }
                var end = DateTime.Now;

                Console.WriteLine("Total time taken: " + (end - start));

            }
            else
            {
                Console.WriteLine("To run this code, provide parameter train, dev, or test, to determine which set to run on");
                Console.WriteLine("This will always train on small training set and run on the chosen small set.");
            }
        }
          

        //read through the file and get posTag set, chunk set, ner tag set, and word set
        private static void getFeatureClassesFromFile(string filePath, out HashSet<string> posSet, out HashSet<string> wordSet, out HashSet<string> chunkSet, out HashSet<string> nerTagSet)
        {
            posSet = new HashSet<string>();
            wordSet = new HashSet<string>();
            chunkSet = new HashSet<string>();
            nerTagSet = new HashSet<string>();

            using (StreamReader reader = new StreamReader(filePath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();

                    while (!string.IsNullOrWhiteSpace(line))
                    {
                        var linesplit = line.Split(' ');

                        wordSet.Add(linesplit[0]);
                        posSet.Add(linesplit[1]);
                        chunkSet.Add(linesplit[2]);
                        nerTagSet.Add(linesplit[3]);
                        line = reader.ReadLine();
                    }
                }
            }

        }

        private static void evaluate(string filePath, string outputPath, Dictionary<string, double> trainedWeights, HashSet<string> tagset)
        {
            var sentenceList = new List<Sentence>();
            using (StreamReader reader = new StreamReader(filePath))
            {

                while (!reader.EndOfStream)
                {
                    var currentSentence = new Sentence();

                    var firstword = new Word();
                    firstword.text = startWord;
                    firstword.posTag = startWord;
                    firstword.chunkTag = startWord;
                    firstword.truth = startWord;
                    currentSentence.words.Add(firstword);

                    var line = reader.ReadLine();
                    while (!string.IsNullOrWhiteSpace(line))
                    {
                        //we have line for word
                        var word = new Word();

                        var lineSplit = line.Split(' ');
                        word.text = lineSplit[0];
                        word.posTag = lineSplit[1];
                        word.chunkTag = lineSplit[2];
                        word.truth = lineSplit[3];

                        currentSentence.words.Add(word);
                        //fill in word object
                        line = reader.ReadLine();
                    }
                    sentenceList.Add(currentSentence);
                }
            }

            using (StreamWriter writer = new StreamWriter(outputPath))
            {
                for (int i = 0; i < sentenceList.Count; i++)
                {
                    if ((i + 1) % 100 == 0)
                    {
                        Console.WriteLine("Evaluating sentence: " + (i + 1));
                    }

                    var currentSentence = sentenceList[i];

                    var tagSequence = Viterbi(currentSentence, trainedWeights, tagset);//we assume tag sequence starts with start

                    var resLine = "";
                    for (int j = 1; j < currentSentence.words.Count; j++)
                    {
                        var word = currentSentence.words[j];
                        resLine = String.Join(" ", word.text, word.posTag, word.chunkTag, word.truth, tagSequence[j]);
                        writer.WriteLine(resLine);
                    }
                    writer.WriteLine("");

                }
            }

        }


        public static Dictionary<string, double> train(string filePath, int numIterations, Dictionary<string, double> initialWeights, HashSet<string> tagset)
        {
            var sentenceList = new List<Sentence>();
            using (StreamReader reader = new StreamReader(filePath))
            {

                while (!reader.EndOfStream)
                {
                    var currentSentence = new Sentence();

                    var firstword = new Word();
                    firstword.text = startWord;
                    firstword.posTag = startWord;
                    firstword.chunkTag = startWord;
                    firstword.truth = startWord;
                    currentSentence.words.Add(firstword);

                    var line = reader.ReadLine();
                    while (!string.IsNullOrWhiteSpace(line))
                    {
                        //we have line for word
                        var word = new Word();

                        var lineSplit = line.Split(' ');
                        word.text = lineSplit[0];
                        word.posTag = lineSplit[1];
                        word.chunkTag = lineSplit[2];
                        word.truth = lineSplit[3];

                        currentSentence.words.Add(word);
                        //fill in word object
                        line = reader.ReadLine();
                    }
                    sentenceList.Add(currentSentence);
                }
            }   //have sentence list

            var currentWeights = new Dictionary<string, double>(initialWeights);//have to actually pre initialize this. adding as time goes will ruin the avg division. one time seeing bigram will add/remove weight of 1time/numsentences not 1/1
            var sumWeights = new Dictionary<string, double>(currentWeights);
            var numWeightsSummed = 1;

            //loop through sentences numIteration times
            for (int iteration = 0; iteration < numIterations; iteration++)
            {
                Console.WriteLine("Training iteration: " + (iteration + 1));

                //loop through sentence
                for (int i = 0; i < sentenceList.Count; i++)
                {
                    if ((i + 1) % 1000 == 0)
                    {
                        Console.WriteLine("Running Viterbi on sentence: " + (i + 1));
                    }

                    var currentSentence = sentenceList[i];

                    var tagSequence = Viterbi(currentSentence, currentWeights, tagset);//we assume tag sequence starts with start


                    //update weights
                    var predictedFeatureVector = new Dictionary<string, double>();
                    var truthFeatureVector = new Dictionary<string, double>();
                    for (int j = 1; j < currentSentence.words.Count; j++)
                    {
                        var currentWord = currentSentence.words[j];
                        var prevWord = currentSentence.words[j - 1];

                        var wordpredictedFeatures = getFeatures(currentSentence, j, tagSequence[j], tagSequence[j - 1]);
                        predictedFeatureVector = addVectors(predictedFeatureVector, wordpredictedFeatures);

                        var wordTruthFeatures = getFeatures(currentSentence, j, currentWord.truth, prevWord.truth);
                        truthFeatureVector = addVectors(truthFeatureVector, wordTruthFeatures);

                    }

                    //update weights if vectors are not the same 
                    if (!sameVector(predictedFeatureVector, truthFeatureVector))
                    {
                        currentWeights = addVectorsWeight(currentWeights, truthFeatureVector);
                        currentWeights = subtractVectors(currentWeights, predictedFeatureVector);

                        sumWeights = addVectorsWeight(sumWeights, currentWeights);
                        numWeightsSummed++;
                    }

                }
            }

            var finalWeightVector = divideVector(sumWeights, numWeightsSummed);
            return finalWeightVector;

        }

        //check if 2 vectors are the same. double comparison is ok since we start with same weight vector and add integers from features
        private static bool sameVector(Dictionary<string, double> predictedFeatureVector, Dictionary<string, double> truthFeatureVector)
        {
            if (predictedFeatureVector.Count != truthFeatureVector.Count)
            {
                return false;
            }
            foreach (var tag in predictedFeatureVector)
            {
                if (truthFeatureVector.ContainsKey(tag.Key))
                {
                    if (truthFeatureVector[tag.Key] == tag.Value)
                    {

                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }
            return true;

        }

        //used for getting the avg weight vector, divide each item in the dictionary with the denominator
        private static Dictionary<string, double> divideVector(Dictionary<string, double> vector, int denominator)
        {
            var finalDiv = new Dictionary<string, double>();
            foreach (var feature in vector)
            {
                finalDiv.Add(feature.Key, feature.Value / denominator);
            }

            return finalDiv;
        }

        //we assume the first item is the weight vector, we only need to loop through smaller list to add it
        private static Dictionary<string, double> addVectorsWeight(Dictionary<string, double> mainVector, Dictionary<string, double> secondVector)
        {
            foreach (var secondFeature in secondVector)
            {
                mainVector[secondFeature.Key] += secondFeature.Value;
            }
            return mainVector;
        }

        //add 2 vecotrs where one isnt the weight. need to make sure items in each list are represented
        private static Dictionary<string, double> addVectors(Dictionary<string, double> mainVector, Dictionary<string, double> secondVector)
        {
            var finalSum = new Dictionary<string, double>();
            foreach (var feature in mainVector)
            {
                var key = feature.Key;
                var value = feature.Value;
                var secondValue = secondVector.ContainsKey(key) ? secondVector[key] : 0.0;

                finalSum.Add(key, value + secondValue);
            }

            foreach (var secondFeature in secondVector)
            {

                if (!finalSum.ContainsKey(secondFeature.Key))
                {
                    finalSum.Add(secondFeature.Key, secondFeature.Value);
                }
            }
            return finalSum;
        }

        //we assume that the main vector is a weight vector and so we only need to subract the second vector
        private static Dictionary<string, double> subtractVectors(Dictionary<string, double> mainVector, Dictionary<string, double> secondVector)
        {
            foreach (var secondFeature in secondVector)
            {
                mainVector[secondFeature.Key] -= secondFeature.Value;

            }
            return mainVector;
        }

        //sets the appropriate features to 1 if they are present
        public static Dictionary<string, double> getFeatures(Sentence sentence, int wordNumber, string currentTag, string previousTag)
        {
            Dictionary<string, double> featureVector = new Dictionary<string, double>();

            //handle stop
            if (currentTag.Equals(stopWord))
            {
                featureVector.Add(previousTag + "#" + currentTag, 1);
                return featureVector;
            }

            var currentWord = sentence.words[wordNumber];

            //first letter is uppercase
            if (char.IsUpper(currentWord.text[0]))
            {
                featureVector.Add("firstUpper" + currentTag, 1);
            }

            //whole word is upper case
            if (currentWord.text.Equals(currentWord.text.ToUpper()))
            {
                featureVector.Add("allUpper" + currentTag, 1);
            }

            //current tag
            featureVector.Add("NER#" + currentTag, 1);

            //bigram of prediction class 
            featureVector.Add(previousTag + "#" + currentTag, 1);

            // word feature
            featureVector.Add(currentTag+"#"+currentWord.text, 1);

            return featureVector;

        }

        //sets our initial weight vector
        private static Dictionary<string, double> initializeWeights(HashSet<string> tagset, HashSet<string> posSet, HashSet<string> wordSet, HashSet<string> chunkSet)
        {

            var weights = new Dictionary<string, double>();

            foreach (var outside in tagset)
            {
                foreach (var inside in tagset)
                {
                    //add all bigrams
                    weights.Add(outside + "#" + inside, rand.NextDouble());
                }

                weights.Add("firstUpper" + outside, rand.NextDouble());
                weights.Add("allUpper" + outside, rand.NextDouble());

                //add start-tag bigrams
                weights.Add(startWord + "#" + outside, rand.NextDouble());
                //add tag-stop bigrams
                weights.Add(outside + "#" + stopWord, rand.NextDouble());
                //add unigram tag
                weights.Add("NER#" + outside, rand.NextDouble());

                //WORD CLASS FEATURE
                foreach (var word in wordSet)
                {
                    weights.Add(outside+"#" + word, rand.NextDouble());
                }
            }

            return weights;
        }

        public static List<string> Viterbi(Sentence sentence, Dictionary<String, double> weights, HashSet<string> stateSet)
        {

            var bestScore = new Dictionary<string, double>();
            var bestTag = new Dictionary<string, string>();

            //first word 
            int i = 1;
            foreach (var tag in stateSet)
            {
                var key = i + " " + tag;
                var featureVector = getFeatures(sentence, i, tag, startWord);
                var score = dotProduct(weights, featureVector);

                bestScore.Add(key, score);
                bestTag.Add(key, startWord);
            }

            //middle
            for (i = 2; i < sentence.words.Count; i++)
            {
                foreach (var tag in stateSet)
                {
                    foreach (var prevTag in stateSet)
                    {
                        var bestPreviousKey = (i - 1) + " " + prevTag;
                        var bestPreviousValue = bestScore[bestPreviousKey];
                        var key = i + " " + tag;
                        var featureVector = getFeatures(sentence, i, tag, prevTag);
                        var score = bestPreviousValue + dotProduct(weights, featureVector);

                        if (bestScore.ContainsKey(key))
                        {
                            var currentScore = bestScore[key];
                            if (score > currentScore)
                            {
                                bestScore[key] = score;
                                bestTag[key] = bestPreviousKey;
                            }
                        }
                        else
                        {
                            bestScore.Add(key, score);
                            bestTag.Add(key, bestPreviousKey);
                        }
                    }
                }
            }

            //last one

            foreach (var prevTag in stateSet)
            {
                var bestPreviousKey = (i - 1) + " " + prevTag;
                var bestPreviousValue = bestScore[bestPreviousKey];


                var key = "" + i + " " + stopWord;
                var featureVector = getFeatures(sentence, i, stopWord, prevTag);
                var score = bestPreviousValue + dotProduct(weights, featureVector);

                if (bestScore.ContainsKey(key))
                {
                    var currentScore = bestScore[key];
                    if (score > currentScore)
                    {
                        bestScore[key] = score;
                        bestTag[key] = bestPreviousKey;
                    }
                }
                else
                {
                    bestScore.Add(key, score);
                    bestTag.Add(key, bestPreviousKey);
                }
            }

            //backtrack and give tags

            List<string> tags = new List<string>();

            var currentTag = i + " " + stopWord;
            while (!currentTag.Equals(startWord))
            {
                var tagsplit = bestTag[currentTag].Split(' ');
                if (tagsplit.Length == 1)
                {
                    tags.Insert(0, tagsplit[0]);
                }
                else
                {
                    tags.Insert(0, tagsplit[1]);
                }
                currentTag = bestTag[currentTag];
            }

            return tags;
        }

        private static double dotProduct(Dictionary<string, double> mainv1, Dictionary<string, double> v2)
        {
            var sum = 0.0;
            foreach (var cell in v2)
            {
                if (mainv1.ContainsKey(cell.Key))
                {
                    sum += mainv1[cell.Key] * cell.Value;
                }
            }
            return sum;
        }
    }

    public class Word
    {
        public string text = "";
        public string posTag = "";
        public string chunkTag = "";
        public string truth = "";
    }

    public class Sentence
    {
        public List<Word> words = new List<Word>();
    }
}
