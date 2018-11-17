using System;
using System.Collections.Generic;
using System.IO;

namespace hw3
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            var trainDataPath = @"../../conll03_ner/eng.train.small";

            train(trainDataPath);


        }

        public static void train(string filePath){

            using (StreamReader reader = new StreamReader(filePath))
            {
                var list = new List<Sentence>();
                while (!reader.EndOfStream)
                {
                    var currentSentence = new Sentence();
                    var line = reader.ReadLine();
                    while(!string.IsNullOrWhiteSpace(line))
                    {
                        //we have line for word
                        var word = new Word();

                        var lineSplit = line.Split(' ');
                        word
                        //fill in word object
                        line = reader.ReadLine();
                    }
                    list.Add(currentSentence);
                }
            }
        }
    }

    public class Word
    {
        public string text = "";
        public string postTag = "";
        public string chunkTag = "";
        public string truth = "";
    }

    public class Sentence
    {
        public List<Word> words = new List<Word>();
    }
}
