using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace hw2
{
    internal class MainClass
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            Dictionary<String, Dictionary<String, int>> transitionCounts = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<String, Dictionary<String, int>> emissionCounts = new Dictionary<string, Dictionary<string, int>>();
            HashSet<string> stateSet = new HashSet<string>();

            //read train data
            var trainDataPath = @"../../CSE517_HW_HMM_Data/twt.train.json";
            using (StreamReader reader = new StreamReader(trainDataPath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var obj = JsonConvert.DeserializeObject<List<List<string>>>(line);
                    obj.Insert(0, new List<string> { "START", "START" });
                    for (int i = 0; i < obj.Count - 1; i++)
                    {

                        var word = obj[i][0];
                        var tag = obj[i][1];
                        var nextTag = obj[i + 1][1];

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
                            var wordDictionary = new Dictionary<string, int>();
                            wordDictionary.Add(word, 1);
                            emissionCounts.Add(tag, wordDictionary);
                        }

                        //transition

                        if (transitionCounts.ContainsKey(tag))
                        {
                            var tagValues = transitionCounts[tag];
                            if (tagValues.ContainsKey(nextTag))
                            {
                                tagValues[nextTag]++;
                            }
                            else
                            {
                                tagValues.Add(nextTag, 1);
                            }
                        }
                        else
                        {
                            var tagDictionary = new Dictionary<string, int>();
                            tagDictionary.Add(nextTag, 1);
                            transitionCounts.Add(tag, tagDictionary);
                        }

                    }
                    //handle last on in list
                    var lastWord = obj[obj.Count - 1][0];
                    var lastTag = obj[obj.Count - 1][1];
                    if (emissionCounts.ContainsKey(lastTag))
                    {
                        var tagValues = emissionCounts[lastTag];
                        if (tagValues.ContainsKey(lastWord))
                        {
                            tagValues[lastWord]++;
                        }
                        else
                        {
                            tagValues.Add(lastWord, 1);
                        }
                    }
                    else
                    {
                        var wordDictionary = new Dictionary<string, int>();
                        wordDictionary.Add(lastWord, 1);
                        emissionCounts.Add(lastTag, wordDictionary);
                    }

                    //transition
                    var stopTag = "STOP";
                    if (transitionCounts.ContainsKey(lastTag))
                    {
                        var tagValues = transitionCounts[lastTag];
                        if (tagValues.ContainsKey(stopTag))
                        {
                            tagValues[stopTag]++;
                        }
                        else
                        {
                            tagValues.Add(stopTag, 1);
                        }
                    }
                    else
                    {
                        var tagDictionary = new Dictionary<string, int>();
                        tagDictionary.Add(stopTag, 1);
                        transitionCounts.Add(lastTag, tagDictionary);
                    }
                }
            }
            Console.WriteLine(emissionCounts.Count);
            Console.WriteLine(transitionCounts.Count);

            stateSet.Remove("START");
            Console.WriteLine(stateSet);


            //start the vetrbi 
            forwardViterbi(trainDataPath, transitionCounts, emissionCounts, stateSet);

        }

        public static void forwardViterbi(string dataSet, Dictionary<String, Dictionary<String, int>> transitionCounts, Dictionary<String, Dictionary<String, int>> emissionCounts, HashSet<string> stateSet)
        {
            using (StreamReader reader = new StreamReader(dataSet))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var dataObject = JsonConvert.DeserializeObject<List<List<string>>>(line);

                    //first word 
                    int i = 0;
                    var bestScore = new Dictionary<string, double>();
                    foreach (var tag in stateSet)
                    {
                        var key = "" + i + " " + tag;
                        double prob = logProb(transitionCounts, "START", tag) + logProb(emissionCounts, tag, dataObject[i][0]);
                        bestScore.Add(key, prob);
                    }

                    //middle
                    for (i = 1; i < dataObject.Count - 1; i++)
                    {
                        var word = dataObject[i][0];
                        foreach (var tag in stateSet)
                        {
                            foreach (var prevTag in stateSet)
                            {
                                var bestPreviousKey = "" + (i - 1) + " " + prevTag;
                                var bestPreviousValue = bestScore[bestPreviousKey];
                                var key = "" + i + " " + tag;
                                var logprob = bestPreviousValue + logProb(transitionCounts, prevTag, tag) + logProb(emissionCounts, tag, dataObject[i][0]);

                                if (bestScore.ContainsKey(key))
                                {
                                    var currentlogprob = bestScore[key];
                                    if (logprob < currentlogprob)
                                    {
                                        bestScore[key] = logprob;
                                    }
                                }
                                else
                                {
                                    bestScore.Add(key, logprob);
                                }

                            }
                        }
                    }

                    //last one

                }
            }
        }

        public static double logProb(Dictionary<String, Dictionary<String, int>> dict, string first, string second)
        {
            double numerator = dict[first][second];
            double denominator = dict[first].Sum(x => x.Value);
            double division = numerator / denominator;
            double log = Math.Log(division);
            return log;
        }
    }
}
