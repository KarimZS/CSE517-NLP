using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace hw2
{
    internal class MainClass
    {
        public static string unkWord = "*UNK*";
        public static string startWord = "*START*";
        public static string stopWord = "*STOP*";

        public static int unkCount = 2;
        public static double k = .001;
        public static double sentenceCount = 0;


        public static Dictionary<string, int> wordSet;
        public static void Main(string[] args)
        {
            if(args.Length<2)
            {
                Console.WriteLine("Run with paramter bigram or trigram then specify train, dev, or test, to determine which set to test on (Ex: trigram dev to get results for trigram hmm results on dev set). All training is done with training set");
                return;
            }

            var trainDataPath = @"../../CSE517_HW_HMM_Data/twt.train.json";
            var filteredTraininDataPath = @"../../CSE517_HW_HMM_Data/twt.train.filtered.json";
            wordSet = handleUnkown(trainDataPath, filteredTraininDataPath, unkCount);
            handleUnkown(trainDataPath, filteredTraininDataPath, unkCount);
            
            switch (args[0])
            {
                case "bigram":

                    //read train data

                    train(trainDataPath, out HashSet<string> stateSet, out Dictionary<string, Dictionary<string, double>> emissionCounts, out Dictionary<string, Dictionary<string, double>> transitionCounts);

                    //smooth emissions add -k 
                    var newEmission = addkSmooth(emissionCounts);
                    var newTransition = addkSmooth(transitionCounts);

                    //convert to probabilities
                    var probEmission = convertToProbability(newEmission);
                    var probTransition = convertToProbability(newTransition);

                    switch (args[1])
                    {
                        case "train":
                            //start the vetrbi 
                            string outputPath = @"../../Results/twt.train.results.json";
                            Viterbi(trainDataPath, outputPath, probTransition, probEmission, stateSet);

                            var confusionResult = createConfusionMatrix(trainDataPath, outputPath);

                            var confusionOutput = @"../../Results/twt.train.confusion.txt";
                            var finalResult = writeConfusionMatrix(confusionOutput, confusionResult.Item2, confusionResult.Item1);

                            var totalCorrect = finalResult.Item1;
                            var totalTags = finalResult.Item2;

                            Console.WriteLine("Correct:" + totalCorrect + " Total:" + totalTags + " Percent:" + (totalCorrect / totalTags));

                            break;
                        case "dev":
                            var devDataPath = @"../../CSE517_HW_HMM_Data/twt.dev.json";
                            //start the vetrbi 
                            string devoutputPath = @"../../Results/twt.dev.results.json";
                            Viterbi(devDataPath, devoutputPath, probTransition, probEmission, stateSet);

                            var devconfusionResult = createConfusionMatrix(devDataPath, devoutputPath);

                            var devconfusionOutput = @"../../Results/twt.dev.confusion.txt";
                            var devfinalResult = writeConfusionMatrix(devconfusionOutput, devconfusionResult.Item2, devconfusionResult.Item1);

                            var devtotalCorrect = devfinalResult.Item1;
                            var devtotalTags = devfinalResult.Item2;

                            Console.WriteLine("Correct:" + devtotalCorrect + " Total:" + devtotalTags + " Percent:" + (devtotalCorrect / devtotalTags));

                            break;
                        case "test":
                            var testDataPath = @"../../CSE517_HW_HMM_Data/twt.test.json";
                            //start the vetrbi 
                            string testoutputPath = @"../../Results/twt.test.results.json";
                            Viterbi(testDataPath, testoutputPath, probTransition, probEmission, stateSet);

                            var testconfusionResult = createConfusionMatrix(testDataPath, testoutputPath);

                            var testconfusionOutput = @"../../Results/twt.test.confusion.txt";
                            var testfinalResult = writeConfusionMatrix(testconfusionOutput, testconfusionResult.Item2, testconfusionResult.Item1);

                            var testtotalCorrect = testfinalResult.Item1;
                            var testtotalTags = testfinalResult.Item2;

                            Console.WriteLine("Correct:" + testtotalCorrect + " Total:" + testtotalTags + " Percent:" + (testtotalCorrect / testtotalTags));

                            break;
                        default:
                            Console.WriteLine("run with paramter train, dev, or test, to determine which set to test on. All training is done with training set");
                            break;
                    }

                    break;
                case "trigram":

                    //read train data

                    tritrain(trainDataPath, out HashSet<string> tristateSet, out Dictionary<string, Dictionary<string, double>> triemissionCounts, out Dictionary<string, Dictionary<string, Dictionary<string, double>>> tritransitionCounts);

                    //smooth emissions add -k 
                    var trinewEmission = addkSmooth(triemissionCounts);
                    var trinewTransition = triaddkSmooth(tritransitionCounts);

                    //convert to probabilities
                    var triprobEmission = convertToProbability(trinewEmission);
                    var triprobTransition = triconvertToProbability(trinewTransition);

                    switch (args[1])
                    {
                        case "train":
                            //start the vetrbi 
                            string outputPath = @"../../Results/twt.train.tri.results.json";
                            triViterbi(trainDataPath, outputPath, triprobTransition, triprobEmission, tristateSet);

                            var confusionResult = createConfusionMatrix(trainDataPath, outputPath);

                            var confusionOutput = @"../../Results/twt.train.tri.confusion.txt";
                            var finalResult = writeConfusionMatrix(confusionOutput, confusionResult.Item2, confusionResult.Item1);

                            var totalCorrect = finalResult.Item1;
                            var totalTags = finalResult.Item2;

                            Console.WriteLine("Correct:" + totalCorrect + " Total:" + totalTags + " Percent:" + (totalCorrect / totalTags));

                            break;
                        case "dev":
                            var devDataPath = @"../../CSE517_HW_HMM_Data/twt.dev.json";
                            //start the vetrbi 
                            string devoutputPath = @"../../Results/twt.dev.tri.results.json";
                            triViterbi(devDataPath, devoutputPath, triprobTransition, triprobEmission, tristateSet);

                            var devconfusionResult = createConfusionMatrix(devDataPath, devoutputPath);

                            var devconfusionOutput = @"../../Results/twt.dev.tri.confusion.txt";
                            var devfinalResult = writeConfusionMatrix(devconfusionOutput, devconfusionResult.Item2, devconfusionResult.Item1);

                            var devtotalCorrect = devfinalResult.Item1;
                            var devtotalTags = devfinalResult.Item2;

                            Console.WriteLine("Correct:" + devtotalCorrect + " Total:" + devtotalTags + " Percent:" + (devtotalCorrect / devtotalTags));

                            break;
                        case "test":
                            var testDataPath = @"../../CSE517_HW_HMM_Data/twt.test.json";
                            //start the vetrbi 
                            string testoutputPath = @"../../Results/twt.test.tri.results.json";
                            triViterbi(testDataPath, testoutputPath, triprobTransition, triprobEmission, tristateSet);

                            var testconfusionResult = createConfusionMatrix(testDataPath, testoutputPath);

                            var testconfusionOutput = @"../../Results/twt.test.tri.confusion.txt";
                            var testfinalResult = writeConfusionMatrix(testconfusionOutput, testconfusionResult.Item2, testconfusionResult.Item1);

                            var testtotalCorrect = testfinalResult.Item1;
                            var testtotalTags = testfinalResult.Item2;

                            Console.WriteLine("Correct:" + testtotalCorrect + " Total:" + testtotalTags + " Percent:" + (testtotalCorrect / testtotalTags));

                            break;
                        default:
                            Console.WriteLine("Run with paramter bigram or trigram then specify train, dev, or test, to determine which set to test on (Ex: trigram dev to get results for trigram hmm results on dev set). All training is done with training set");
                            break;
                    }
                    break;
                default:
                    Console.WriteLine("Run with paramter bigram or trigram then specify train, dev, or test, to determine which set to test on (Ex: trigram dev to get results for trigram hmm results on dev set). All training is done with training set");
                    break;
            }


        }

        private static void tritrain(string trainDataPath, out HashSet<string> tristateSet, out Dictionary<string, Dictionary<string, double>> triemissionCounts, out Dictionary<string, Dictionary<string, Dictionary<string, double>>> tritransitionCounts)
        {
            tritransitionCounts = new Dictionary<string, Dictionary<string, Dictionary<string, double>>>();
            triemissionCounts = new Dictionary<string, Dictionary<string, double>>();
            tristateSet = new HashSet<string>();

            using (StreamReader reader = new StreamReader(trainDataPath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var obj = JsonConvert.DeserializeObject<List<List<string>>>(line);
                    obj.Insert(0, new List<string> { startWord, startWord });
                    obj.Insert(0, new List<string> { startWord, startWord });
                    for (int i = 2; i < obj.Count; i++)
                    {

                        var word = obj[i][0];
                        var tag = obj[i][1];
                        var prevTag = obj[i - 1][1];
                        var twoPrevTag = obj[i - 2][1];

                        tristateSet.Add(tag);

                        //emission
                        if (triemissionCounts.ContainsKey(tag))
                        {
                            var tagValues = triemissionCounts[tag];
                            if (tagValues.ContainsKey(word))
                            {
                                tagValues[word]++;
                            }
                            else
                            {
                                tagValues.Add(word, 1);
                            }
                        }
                        else
                        {
                            var wordDictionary = new Dictionary<string, double>();
                            wordDictionary.Add(word, 1);
                            triemissionCounts.Add(tag, wordDictionary);
                        }

                        //transition

                        if (tritransitionCounts.ContainsKey(twoPrevTag))
                        {
                            var twoprevTagValues = tritransitionCounts[twoPrevTag];
                            if (twoprevTagValues.ContainsKey(prevTag))
                            {
                                var prevTagValue = twoprevTagValues[prevTag];
                                if (prevTagValue.ContainsKey(tag))
                                {
                                    prevTagValue[tag]++;
                                }
                                else
                                {
                                    prevTagValue.Add(tag, 1);
                                }
                            }
                            else
                            {
                                var tagDictionary = new Dictionary<string, double>();
                                tagDictionary.Add(tag, 1);
                                twoprevTagValues.Add(prevTag, tagDictionary);
                            }
                        }
                        else
                        {
                            var tagDictionary = new Dictionary<string, double>();
                            tagDictionary.Add(tag, 1);
                            var twotagDictionary = new Dictionary<string, Dictionary<string, double>>();
                            twotagDictionary.Add(prevTag, tagDictionary);
                            tritransitionCounts.Add(twoPrevTag, twotagDictionary);
                        }

                    }
                    //handle last transition to stop in list


                    var twoPrev = obj[obj.Count - 2][1];
                    var prev = obj.Last()[1];

                    if (tritransitionCounts.ContainsKey(twoPrev))
                    {
                        var twoprevTagValues = tritransitionCounts[twoPrev];
                        if (twoprevTagValues.ContainsKey(prev))
                        {
                            var prevTagValue = twoprevTagValues[prev];
                            if (prevTagValue.ContainsKey(stopWord))
                            {
                                prevTagValue[stopWord]++;
                            }
                            else
                            {
                                prevTagValue.Add(stopWord, 1);
                            }
                        }
                        else
                        {
                            var tagDictionary = new Dictionary<string, double>();
                            tagDictionary.Add(stopWord, 1);
                            twoprevTagValues.Add(prev, tagDictionary);
                        }
                    }
                    else
                    {
                        var tagDictionary = new Dictionary<string, double>();
                        tagDictionary.Add(stopWord, 1);
                        var twotagDictionary = new Dictionary<string, Dictionary<string, double>>();
                        twotagDictionary.Add(prev, tagDictionary);
                        tritransitionCounts.Add(twoPrev, twotagDictionary);
                    }

                    //update sentence count = # stop tags
                    sentenceCount++;
                }
            }

            tristateSet.Remove(startWord);
        }

        public static void train(string trainDataPath, out HashSet<string> stateSet, out Dictionary<String, Dictionary<String, double>> emissionCounts, out Dictionary<String, Dictionary<String, double>> transitionCounts)
        {
            transitionCounts = new Dictionary<string, Dictionary<string, double>>();
            emissionCounts = new Dictionary<string, Dictionary<string, double>>();
            stateSet = new HashSet<string>();

            using (StreamReader reader = new StreamReader(trainDataPath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var obj = JsonConvert.DeserializeObject<List<List<string>>>(line);
                    obj.Insert(0, new List<string> { startWord, startWord });
                    for (int i = 1; i < obj.Count; i++)
                    {

                        var word = obj[i][0];
                        var tag = obj[i][1];
                        var prevTag = obj[i - 1][1];

                        stateSet.Add(tag);

                        //emission
                        if (emissionCounts.ContainsKey(tag))
                        {
                            var tagValues = emissionCounts[tag];
                            if (tagValues.ContainsKey(word))
                            {
                                tagValues[word]++;
                            }
                            else
                            {
                                tagValues.Add(word, 1);
                            }
                        }
                        else
                        {
                            var wordDictionary = new Dictionary<string, double>();
                            wordDictionary.Add(word, 1);
                            emissionCounts.Add(tag, wordDictionary);
                        }

                        //transition

                        if (transitionCounts.ContainsKey(prevTag))
                        {
                            var tagValues = transitionCounts[prevTag];
                            if (tagValues.ContainsKey(tag))
                            {
                                tagValues[tag]++;
                            }
                            else
                            {
                                tagValues.Add(tag, 1);
                            }
                        }
                        else
                        {
                            var tagDictionary = new Dictionary<string, double>();
                            tagDictionary.Add(tag, 1);
                            transitionCounts.Add(prevTag, tagDictionary);
                        }

                    }
                    //handle last transition to stop in list

                    if (transitionCounts.ContainsKey(obj.Last()[1]))
                    {
                        var tagValues = transitionCounts[obj.Last()[1]];
                        if (tagValues.ContainsKey(stopWord))
                        {
                            tagValues[stopWord]++;
                        }
                        else
                        {
                            tagValues.Add(stopWord, 1);
                        }
                    }
                    else
                    {
                        var tagDictionary = new Dictionary<string, double>();
                        tagDictionary.Add(stopWord, 1);
                        transitionCounts.Add(obj.Last()[1], tagDictionary);
                    }

                    //update sentence count = # stop tags
                    sentenceCount++;
                }
            }

            stateSet.Remove(startWord);
        }

        public static Dictionary<string, int> handleUnkown(string dataPath, string outputPath, int unk)
        {
            Dictionary<string, int> wordSetL = new Dictionary<string, int>();

            using (StreamReader reader = new StreamReader(dataPath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var obj = JsonConvert.DeserializeObject<List<List<string>>>(line);
                    foreach (var pair in obj)
                    {
                        if (wordSetL.ContainsKey(pair[0]))
                        {
                            wordSetL[pair[0]] += 1;
                        }
                        else
                        {
                            wordSetL.Add(pair[0], 1);
                        }
                    }
                }
            }

            Dictionary<string, int> filteredWordSet = wordSetL.Where(x => (x.Value >= unk)).ToDictionary(x => x.Key, x => x.Value);
            filteredWordSet.Add(unkWord, 0);

            using (StreamReader reader = new StreamReader(dataPath))
            {
                using (StreamWriter writer = new StreamWriter(outputPath))
                {
                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        var obj = JsonConvert.DeserializeObject<List<List<string>>>(line);
                        foreach (var pair in obj)
                        {
                            if (!filteredWordSet.ContainsKey(pair[0]))
                            {
                                pair[0] = unkWord;
                                filteredWordSet[unkWord]++;
                            }
                        }
                        var seralized = JsonConvert.SerializeObject(obj);
                        writer.WriteLine(seralized);
                    }
                }

            }
            return filteredWordSet;
        }

        public static Tuple<double, double> writeConfusionMatrix(string confusionOutput, HashSet<string> completeTagSet, Dictionary<string, Dictionary<string, int>> confusionMatrix)
        {
            double totalCorrect = 0;
            double totalTags = 0;
            List<string> lines = new List<string>();
            var header = "     ";
            foreach (var tag in completeTagSet)
            {
                header += tag + "    ";
                var line = tag + "    ";
                foreach (var secondTag in completeTagSet)
                {
                    if (confusionMatrix.ContainsKey(tag))
                    {
                        var expectedlist = confusionMatrix[tag];
                        if (expectedlist.ContainsKey(secondTag))
                        {
                            if (tag == secondTag)
                            {
                                totalCorrect += confusionMatrix[tag][secondTag];
                            }
                            totalTags += confusionMatrix[tag][secondTag];
                            var value = confusionMatrix[tag][secondTag] + "";
                            var count = 5 - value.Length;
                            for (int l = 0; l < count; l++)
                            {
                                value += " ";
                            }
                            line += value;
                        }
                        else
                        {
                            line += 0 + "    ";
                        }
                    }
                    else
                    {
                        line += 0 + "    ";
                    }
                }
                lines.Add(line);
            }
            using (StreamWriter writer = new StreamWriter(confusionOutput))
            {
                writer.WriteLine(header);
                foreach (var line in lines)
                {
                    writer.WriteLine(line);
                }
            }

            return Tuple.Create(totalCorrect, totalTags);
        }

        public static Tuple<Dictionary<string, Dictionary<string, int>>, HashSet<string>> createConfusionMatrix(string truthFile, string resultFile)
        {
            var confusionMatrix = new Dictionary<string, Dictionary<string, int>>();
            var completeTagSet = new HashSet<string>();

            using (StreamReader expectedReader = new StreamReader(truthFile))
            {
                using (StreamReader predictedReader = new StreamReader(resultFile))
                {
                    while (!predictedReader.EndOfStream)
                    {
                        var expectedLine = expectedReader.ReadLine();
                        var expectedObj = JsonConvert.DeserializeObject<List<List<string>>>(expectedLine);
                        var predictedLine = predictedReader.ReadLine();
                        var predictedObj = JsonConvert.DeserializeObject<List<List<string>>>(predictedLine);

                        for (int i = 0; i < expectedObj.Count; i++)
                        {
                            var expectedTag = expectedObj[i][1];
                            var predictedTag = predictedObj[i][1];

                            completeTagSet.Add(expectedTag);
                            completeTagSet.Add(predictedTag);

                            if (confusionMatrix.ContainsKey(expectedTag))
                            {
                                var expectedlist = confusionMatrix[expectedTag];
                                if (expectedlist.ContainsKey(predictedTag))
                                {
                                    expectedlist[predictedTag]++;
                                }
                                else
                                {
                                    expectedlist.Add(predictedTag, 1);
                                }
                            }
                            else
                            {
                                var temp = new Dictionary<string, int>();
                                temp.Add(predictedTag, 1);
                                confusionMatrix.Add(expectedTag, temp);
                            }
                        }
                    }
                }
            }

            return Tuple.Create(confusionMatrix, completeTagSet);
        }

        public static Dictionary<String, Dictionary<String, Dictionary<String, double>>> triaddkSmooth(Dictionary<String, Dictionary<String, Dictionary<String, double>>> dictionary)
        {

            foreach (var twoprevtag in dictionary)
            {
                var twoprevVal = twoprevtag.Value;
                // var temp = new Dictionary<string, Dictionary<string, double>>();
                var innerTemp = new Dictionary<string, double>();
                // innerTemp.Add(unkWord, 0);
                //temp.Add(unkWord, innerTemp);
                twoprevVal.Add(unkWord, innerTemp);

                foreach (var prevtag in twoprevVal)
                {
                    var prevval = prevtag.Value;
                    prevval.Add(unkWord, 0);
                }
            }
            var newCount = new Dictionary<String, Dictionary<String, Dictionary<String, double>>>();
            double sum = dictionary.First().Value.First().Value.Count * k;
            foreach (var twoprevtag in dictionary)
            {
                var tempA = new Dictionary<string, Dictionary<string, double>>();

                foreach (var prevtag in twoprevtag.Value)
                {
                    var tempB = new Dictionary<string, double>();

                    foreach (var tag in prevtag.Value)
                    {
                        var rest = dictionary[twoprevtag.Key][prevtag.Key][tag.Key] + (sum / prevtag.Value.Count);
                        tempB.Add(tag.Key, rest);

                    }
                    tempA.Add(prevtag.Key, tempB);
                }
                newCount.Add(twoprevtag.Key, tempA);
            }

            return newCount;
        }

        public static Dictionary<string, Dictionary<string, double>> addkSmooth(Dictionary<String, Dictionary<String, double>> dictionary)
        {
            var newCount = new Dictionary<String, Dictionary<String, double>>();
            foreach (var tag in dictionary)
            {
                if (tag.Value.ContainsKey(unkWord))
                {
                    newCount.Add(tag.Key, tag.Value);
                    continue;
                }
                else
                {
                    var temp = new Dictionary<string, double>();
                    foreach (var emission in tag.Value)
                    {
                        temp.Add(emission.Key, emission.Value + k);
                    }
                    temp.Add(unkWord, k);
                    newCount.Add(tag.Key, temp);
                }

            }

            return newCount;
        }


        public static Dictionary<String, Dictionary<String, double>> convertToProbability(Dictionary<String, Dictionary<String, double>> dictionary)
        {

            var prob = new Dictionary<String, Dictionary<String, double>>();
            foreach (var tag in dictionary)
            {
                var newlist = new Dictionary<string, double>();

                var sum = tag.Value.Sum(x => x.Value);
                foreach (var innertag in tag.Value)
                {
                    double numerator = innertag.Value;
                    double denominator = sum;
                    var lognm = Math.Log(numerator);
                    var logdm = Math.Log(denominator);
                    double logdivision = lognm - logdm;
                    newlist.Add(innertag.Key, logdivision);
                }
                prob.Add(tag.Key, newlist);

            }

            return prob;
        }

        public static Dictionary<String, Dictionary<String, Dictionary<String, double>>> triconvertToProbability(Dictionary<String, Dictionary<String, Dictionary<String, double>>> dictionary)
        {

            var prob = new Dictionary<String, Dictionary<String, Dictionary<String, double>>>();
            foreach (var tag in dictionary)
            {
                var newlist = new Dictionary<string, Dictionary<string, double>>();
                foreach (var innertag in tag.Value)
                {
                    var innerList = new Dictionary<string, double>();
                    var sum = innertag.Value.Sum(x => x.Value);
                    foreach (var finaltag in innertag.Value)
                    {
                        double numerator = finaltag.Value;
                        double denominator = sum;
                        var lognm = Math.Log(numerator);
                        var logdm = Math.Log(denominator);
                        double logdivision = lognm - logdm;
                        innerList.Add(finaltag.Key, logdivision);
                    }
                    newlist.Add(innertag.Key, innerList);

                }


                prob.Add(tag.Key, newlist);

            }

            return prob;
        }

        public static void Viterbi(string dataSet, string outputPath, Dictionary<String, Dictionary<String, double>> transitionCounts, Dictionary<String, Dictionary<String, double>> emissionCounts, HashSet<string> stateSet)
        {
            using (StreamReader reader = new StreamReader(dataSet))
            {
                using (StreamWriter writer = new StreamWriter(outputPath))
                {
                    var linecount = 0;
                    while (!reader.EndOfStream)
                    {
                        linecount++;
                        if (linecount % 1000 == 0)
                        {
                            Console.WriteLine("Running Viterbi on line: " + linecount);
                        }
                        var line = reader.ReadLine();
                        var dataObject = JsonConvert.DeserializeObject<List<List<string>>>(line);
                        var bestScore = new Dictionary<string, double>();
                        var bestTag = new Dictionary<string, string>();

                        //first word 
                        int i = 0;
                        foreach (var tag in stateSet)
                        {
                            var key = "" + i + " " + tag;
                            var word = emissionCounts[tag].ContainsKey(dataObject[i][0]) ? dataObject[i][0] : unkWord;
                            var finTag = transitionCounts[startWord].ContainsKey(tag) ? tag : unkWord;
                            double prob = transitionCounts[startWord][finTag] + emissionCounts[tag][word];
                            bestScore.Add(key, prob);
                            bestTag.Add(key, startWord);
                        }

                        //middle
                        for (i = 1; i < dataObject.Count; i++)
                        {
                            foreach (var tag in stateSet)
                            {
                                var word = emissionCounts[tag].ContainsKey(dataObject[i][0]) ? dataObject[i][0] : unkWord;


                                foreach (var prevTag in stateSet)
                                {
                                    var finTag = transitionCounts[prevTag].ContainsKey(tag) ? tag : unkWord;
                                    var bestPreviousKey = "" + (i - 1) + " " + prevTag;
                                    var bestPreviousValue = bestScore[bestPreviousKey];
                                    var key = "" + i + " " + tag;
                                    var logprob = bestPreviousValue + transitionCounts[prevTag][finTag] + emissionCounts[tag][word];

                                    if (bestScore.ContainsKey(key))
                                    {
                                        var currentlogprob = bestScore[key];
                                        if (logprob > currentlogprob)
                                        {
                                            bestScore[key] = logprob;
                                            bestTag[key] = bestPreviousKey;
                                        }
                                    }
                                    else
                                    {
                                        bestScore.Add(key, logprob);
                                        bestTag.Add(key, bestPreviousKey);
                                    }
                                }
                            }
                        }

                        //last one

                        foreach (var prevTag in stateSet)
                        {
                            var bestPreviousKey = "" + (i - 1) + " " + prevTag;
                            var bestPreviousValue = bestScore[bestPreviousKey];
                            var finTag = transitionCounts[prevTag].ContainsKey(stopWord) ? stopWord : unkWord;

                            var key = "" + i + " " + stopWord;
                            var logprob = bestPreviousValue + transitionCounts[prevTag][finTag];

                            if (bestScore.ContainsKey(key))
                            {
                                var currentlogprob = bestScore[key];
                                if (logprob > currentlogprob)
                                {
                                    bestScore[key] = logprob;
                                    bestTag[key] = bestPreviousKey;
                                }
                            }
                            else
                            {
                                bestScore.Add(key, logprob);
                                bestTag.Add(key, bestPreviousKey);
                            }
                        }

                        //backtrack and give tags

                        List<string> tags = new List<string>();

                        var currentTag = i + " " + stopWord;
                        while (!currentTag.Equals(startWord))
                        {
                            tags.Insert(0, bestTag[currentTag]);
                            currentTag = bestTag[currentTag];
                        }

                        var outobj = new List<List<string>>();

                        for (int m = 1; m <= dataObject.Count; m++)
                        {
                            var tuple = new List<string>();
                            tuple.Add(dataObject[m - 1][0]);
                            tuple.Add(tags[m].Split(' ')[1]);
                            outobj.Add(tuple);
                        }
                        var json = JsonConvert.SerializeObject(outobj);
                        writer.WriteLine(json);
                    }

                }
            }
        }

        public static void triViterbi(string dataSet, string outputPath, Dictionary<String, Dictionary<String, Dictionary<String, double>>> transitionCounts, Dictionary<String, Dictionary<String, double>> emissionCounts, HashSet<string> stateSet)
        {
            using (StreamReader reader = new StreamReader(dataSet))
            {
                using (StreamWriter writer = new StreamWriter(outputPath))
                {
                    var linecount = 0;
                    while (!reader.EndOfStream)
                    {
                        linecount++;
                        if (linecount % 1000 == 0)
                        {
                            Console.WriteLine("Running trigram Viterbi on line: " + linecount);
                        }
                        var line = reader.ReadLine();
                        var dataObject = JsonConvert.DeserializeObject<List<List<string>>>(line);
                        var bestScore = new Dictionary<string, double>();
                        var bestTag = new Dictionary<string, string>();


                        //start start first tag
                        int i = 0;
                        foreach (var tag in stateSet)
                        {
                            var key = "" + i + " " + startWord + " " + tag;
                            var word = emissionCounts[tag].ContainsKey(dataObject[i][0]) ? dataObject[i][0] : unkWord;
                            var prevtag = startWord;
                            var prev2tag = startWord;
                            var finTag = transitionCounts[prev2tag][prevtag].ContainsKey(tag) ? tag : unkWord;
                            double prob = transitionCounts[prev2tag][prevtag][finTag] + emissionCounts[tag][word];
                            bestScore.Add(key, prob);
                            bestTag.Add(key, startWord);

                        }

                        //start first second
                        if(dataObject.Count==1)
                        {
                            continue;
                        }
                        i = 1;
                        foreach (var tag in stateSet)
                        {
                            foreach (var prevTag in stateSet)
                            {

                                var key = "" + i + " " + prevTag + " " + tag;
                                var word = emissionCounts[tag].ContainsKey(dataObject[i][0]) ? dataObject[i][0] : unkWord;
                                var prev2tag = startWord;
                                var finprevtag = transitionCounts[prev2tag].ContainsKey(prevTag) ? prevTag : unkWord;
                                var finTag = transitionCounts[prev2tag][finprevtag].ContainsKey(tag) ? tag : unkWord;
                                double prob = bestScore[i - 1 + " " + startWord + " " + prevTag] + transitionCounts[prev2tag][finprevtag][finTag] + emissionCounts[tag][word];


                                if (bestScore.ContainsKey(key))
                                {
                                    var currentlogprob = bestScore[key];
                                    if (prob > currentlogprob)
                                    {
                                        bestScore[key] = prob;
                                        bestTag[key] = prev2tag;
                                    }
                                }
                                else
                                {
                                    bestScore.Add(key, prob);
                                    bestTag.Add(key, prev2tag);
                                }
                            }
                        }

                        //first second third
                        for (i = 2; i < dataObject.Count; i++)
                        {
                            foreach (var tag in stateSet)
                            {
                                var word = emissionCounts[tag].ContainsKey(dataObject[i][0]) ? dataObject[i][0] : unkWord;


                                foreach (var prevTag in stateSet)
                                {
                                    foreach (var prev2tag in stateSet)
                                    {
                                        var key = "" + i + " " + prevTag + " " + tag;
                                        var finprevtag = transitionCounts[prev2tag].ContainsKey(prevTag) ? prevTag : unkWord;
                                        var finTag = transitionCounts[prev2tag][finprevtag].ContainsKey(tag) ? tag : unkWord;
                                        double prob = bestScore[i - 1 + " " + prev2tag + " " + prevTag] + transitionCounts[prev2tag][finprevtag][finTag] + emissionCounts[tag][word];


                                        if (bestScore.ContainsKey(key))
                                        {
                                            var currentlogprob = bestScore[key];
                                            if (prob > currentlogprob)
                                            {
                                                bestScore[key] = prob;
                                                bestTag[key] = prev2tag;
                                            }
                                        }
                                        else
                                        {
                                            bestScore.Add(key, prob);
                                            bestTag.Add(key, prev2tag);
                                        }
                                    }
                                }
                            }
                        }


                        //transition to stop
                        var bestprev = "";
                        var bestprev2 = "";
                        double finalScore = Double.MinValue;
                        foreach (var prevTag in stateSet)
                        {
                            foreach (var prev2tag in stateSet)
                            {
                                var key = "" + i + " " + stopWord;
                                var finprevtag = transitionCounts[prev2tag].ContainsKey(prevTag) ? prevTag : unkWord;
                                var finTag = transitionCounts[prev2tag][finprevtag].ContainsKey(stopWord) ? stopWord : unkWord;
                                double prob = bestScore[i - 1 + " " + prev2tag + " " + prevTag] + transitionCounts[prev2tag][finprevtag][finTag];

                                if (prob > finalScore)
                                {
                                    finalScore = prob;
                                    bestprev = prevTag;
                                    bestprev2 = prev2tag;
                                }

                            }
                        }


                        //backtrack and give tags

                        List<string> tags = new List<string>();

                        tags.Add(bestprev);
                        tags.Insert(0,bestprev2);
                        i--;//go back to before stop
                        while(i>=0)
                        {

                            var key = i + " " + tags[1] + " " + tags[0];
                            var val = bestTag[key];
                            tags.Insert(0, val);
                            if(val.Equals(startWord))
                            {
                                break;
                            }

                            i--;
                        }

                        tags.RemoveAt(0);

                       var outobj = new List<List<string>>();

                        for (int m = 0; m < dataObject.Count; m++)
                        {
                            var tuple = new List<string>();
                            tuple.Add(dataObject[m][0]);
                            tuple.Add(tags[m]);
                            outobj.Add(tuple);
                        }
                        var json = JsonConvert.SerializeObject(outobj);
                        writer.WriteLine(json);
                    }

                }
            }
        }
    }
}
