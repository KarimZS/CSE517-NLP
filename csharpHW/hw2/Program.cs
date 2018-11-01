using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace hw2
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            Dictionary<String, Dictionary<String, int>> transitionCounts = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<String, Dictionary<String, int>> emissionCounts = new Dictionary<string, Dictionary<string, int>>();


            //read train data
            var trainDataPath = @"../../CSE517_HW_HMM_Data/twt.train.json";
            using (StreamReader reader = new StreamReader(trainDataPath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var obj = JsonConvert.DeserializeObject<List<List<string>>>(line);
                    for (int i = 0; i < obj.Count - 1; i++)
                    {
                        //emission
                        var word = obj[i][0];
                        var tag = obj[i][1];
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
                    }
                }
            }
            Console.WriteLine(emissionCounts.Count);
        }
    }
}
