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
        public static double lambda1 = 0.01;
        public static double lambda2 = 0.99;
        public static double k = 1;
        public static double sentenceCount = 0;


        public static Dictionary<string, int> wordSet;
        public static void Main(string[] args)
        {

            Dictionary<String, Dictionary<String, double>> transitionCounts = new Dictionary<string, Dictionary<string, double>>();
            Dictionary<String, Dictionary<String, double>> emissionCounts = new Dictionary<string, Dictionary<string, double>>();
            HashSet<string> stateSet = new HashSet<string>();


            var trainDataPath = @"../../CSE517_HW_HMM_Data/twt.train.json";
            var filteredTraininDataPath = @"../../CSE517_HW_HMM_Data/twt.train.filtered.json";
            wordSet = handleUnkown(trainDataPath, filteredTraininDataPath, unkCount);

            //read train data

            using (StreamReader reader = new StreamReader(filteredTraininDataPath))
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
                        var prevTag = obj[i -1][1];

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
            Console.WriteLine(emissionCounts.Count);
            Console.WriteLine(transitionCounts.Count);

            stateSet.Remove(startWord);

            //smooth emissions add -k 
            var newEmission = new Dictionary<String, Dictionary<String, double>>();
            foreach (var tag in emissionCounts)
            {
                if (tag.Value.ContainsKey(unkWord))
                {
                    newEmission.Add(tag.Key, tag.Value);
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
                    newEmission.Add(tag.Key, temp);
                }

            }

            //start the vetrbi 
            string outputPath = @"../../Results/twt.train.results.json";
            forwardViterbi(trainDataPath, outputPath, transitionCounts, newEmission, stateSet);

            var confusionMatrix = new Dictionary<string, Dictionary<string, int>>();
            var completeTagSet = new HashSet<string>();

            using (StreamReader expectedReader = new StreamReader(trainDataPath))
            {
                using (StreamReader predictedReader = new StreamReader(outputPath))
                {
                    while (!expectedReader.EndOfStream)
                    {
                        var expectedLine = expectedReader.ReadLine();
                        var expectedObj = JsonConvert.DeserializeObject<List<List<string>>>(expectedLine);
                        var predictedLine = predictedReader.ReadLine();
                        var predictedObj = JsonConvert.DeserializeObject<List<List<string>>>(predictedLine);

                        for (int i = 0; i < expectedObj.Count-1; i++)
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

            var confusionOutput = @"../../Results/twt.train.confusion.txt";
            double totalCorrect = 0;
            double totalTags = 0;
            List<string> lines = new List<string>();
            var header = "\t ";
            foreach (var tag in completeTagSet)
            {
                header += tag + "\t";
                var line = tag + "\t";
                foreach (var secondTag in completeTagSet)
                {
                    totalTags++;
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
                            line += confusionMatrix[tag][secondTag] + "\t";
                        }
                        else
                        {
                            line += 0 + "\t";
                        }
                    }
                    else
                    {
                        line += 0 + "\t";
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

            Console.WriteLine("Correct:"+totalCorrect+" Total:"+totalTags+" Percent:"+(totalCorrect/totalTags));
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

        public static void forwardViterbi(string dataSet, string outputPath, Dictionary<String, Dictionary<String, double>> transitionCounts, Dictionary<String, Dictionary<String, double>> emissionCounts, HashSet<string> stateSet)
        {
            using (StreamReader reader = new StreamReader(dataSet))
            {
                using (StreamWriter writer = new StreamWriter(outputPath))
                {
                    while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var dataObject = JsonConvert.DeserializeObject<List<List<string>>>(line);
                    var bestScore = new Dictionary<string, double>();
                    var bestTag = new Dictionary<string, string>();

                    //first word 
                    int i = 0;
                    foreach (var tag in stateSet)
                    {
                        var key = "" + i + " " + tag;
                        var word = wordSet.ContainsKey(dataObject[i][0]) ? dataObject[i][0] : unkWord;
                        double prob = logProbTransition(transitionCounts, startWord, tag) + logProbEmission(emissionCounts, tag, word);
                        bestScore.Add(key, prob);
                        bestTag.Add(key, startWord);
                    }

                    //middle
                    for (i = 1; i < dataObject.Count; i++)
                    {
                        var word = wordSet.ContainsKey(dataObject[i][0]) ? dataObject[i][0] : unkWord;
                        foreach (var tag in stateSet)
                        {
                            foreach (var prevTag in stateSet)
                            {
                                var bestPreviousKey = "" + (i - 1) + " " + prevTag;
                                var bestPreviousValue = bestScore[bestPreviousKey];
                                var key = "" + i + " " + tag;
                                var logprob = bestPreviousValue + logProbTransition(transitionCounts, prevTag, tag) + logProbEmission(emissionCounts, tag, word);

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
                        var key = "" + i + " " + stopWord;
                        var logprob = bestPreviousValue + logProbTransition(transitionCounts, prevTag, stopWord) + logProbEmission(emissionCounts, stopWord, stopWord);

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
                  
                        for (int m = 1; m < dataObject.Count; m++)
                        {
                            var tuple = new List<string>();
                            tuple.Add(dataObject[m-1][0]);
                            tuple.Add(tags[m].Split(' ')[1]);
                            outobj.Add(tuple);
                        }
                        var json = JsonConvert.SerializeObject(outobj);
                        writer.WriteLine(json);
                    }

                }
            }
        }

        public static double logProbTransition(Dictionary<String, Dictionary<String, double>> dict, string first, string second)
        {
            //bigram
            double numerator = dict[first].ContainsKey(second) ? dict[first][second] : 0;
            double denominator = dict[first].Sum(x => x.Value);
            double logdivision;
            if (numerator == 0)
            {
                logdivision = double.MinValue;
            }
            else
            {
                double lognm = Math.Log(numerator);
                double logdm = Math.Log(denominator);
                logdivision = lognm - logdm;
            }


            //unigram
            double uniNumerator;
            if (second == stopWord)
            {
                uniNumerator = sentenceCount;//times we have seen a stop
            }
            else
            {
                uniNumerator = dict[second].Sum(x => x.Value);
            }
            double uniDenominator = dict.Sum(x => x.Value.Sum(y => y.Value));
            double uninmlog = Math.Log(uniNumerator);
            double unidmlog = Math.Log(uniDenominator);
            double uniDivision = uninmlog - unidmlog;


            return lambda2 * logdivision + lambda1 * uniDivision;
        }

        public static double logProbEmission(Dictionary<String, Dictionary<String, double>> dict, string first, string second)
        {
            if (first == stopWord && second == stopWord)
            {
                return 0;
            }
            //bigram
            double numerator = dict[first].ContainsKey(second) ? dict[first][second] : dict[first][unkWord];
            double denominator = dict[first].Sum(x => x.Value);
            var lognm = Math.Log(numerator);
            var logdm = Math.Log(denominator);
            double logdivision = lognm - logdm;

            ////signle
            //var singleNum = wordSet.ContainsKey(second) ? wordSet[second]: wordSet[unkWord];
            //var sum = wordSet.Sum(x => x.Value);
            //var singleDivision = singleNum / sum;
            //var singleLog = division == 0 ? 0 : Math.Log(division);

            //return lambda2 * log + lambda1 * singleLog;

            return logdivision;
        }
    }
}
