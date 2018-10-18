import java.io.*;
import java.sql.Struct;
import java.util.*;

public class Main {

    public static String startSymbol = "START";
    public static String stopSymbol = "STOP";
    public static String unkSymbol = "UNK";
    public static String keySeparator = "#";

    public static int unkThreshold = 3;


    public static HashMap<String, Integer> unigram_map = null;
    public static HashMap<String, Integer> bigram_map = null;
    public static HashMap<String, Integer> trigram_map = null;
    public static int numLinesInTrain = 0;


    public static void main(String[] args) {
        // write your code here
        String trainFilePath = "CSE517-HW1-Data/prob1_brown_full/brown.train.txt";
        String devFilePath = "CSE517-HW1-Data/prob1_brown_full/brown.dev.txt";
        String testFilePath = "CSE517-HW1-Data/prob1_brown_full/brown.test.txt";


        //get words with count less than 2
        HashMap<String, Integer> words = token_count(trainFilePath);
        HashMap<String, Integer> wordsFiltered = new HashMap<>();

        Set<String> keySet = words.keySet();
        for (String key : keySet) {

            int keyValue = words.get(key);
            if (keyValue >= unkThreshold) {
                wordsFiltered.put(key,keyValue);
            }
        }
        String trainNewFilePath = "CSE517-HW1-Data/prob1_brown_full/brown.train.filtered.txt";
        String devNewFilePath = "CSE517-HW1-Data/prob1_brown_full/brown.dev.filtered.txt";
        String testNewFilePath = "CSE517-HW1-Data/prob1_brown_full/brown.test.filtered.txt";

        //go through file and replace all less than 2 words with UNK
        remove_low_freq_words(trainFilePath,trainNewFilePath,wordsFiltered);
        remove_low_freq_words(devFilePath,devNewFilePath,wordsFiltered);
        remove_low_freq_words(testFilePath,testNewFilePath,wordsFiltered);


        //create 1,2,3 grams
        unigram_map = ngram_count(trainNewFilePath, 1);
        bigram_map = ngram_count(trainNewFilePath, 2);
        trigram_map = ngram_count(trainNewFilePath, 3);

        //run on train,dev,test and give prob of each sentence
        System.out.println("UNIGRAM PROBABILITIES");
        double trainUnigramPerplexity = evaluateSet(trainNewFilePath,1);
        System.out.println(trainUnigramPerplexity);
        double devUnigramPerplexity = evaluateSet(devNewFilePath,1);
        System.out.println(devUnigramPerplexity);
        double testUnigramPerplexity = evaluateSet(testNewFilePath,1);
        System.out.println(testUnigramPerplexity);

        //run on train,dev,test and give prob of each sentence
        System.out.println("BIGRAM PROBABILITIES");
        double trainBigramPerplexity = evaluateSet(trainNewFilePath,2);
        System.out.println(trainBigramPerplexity);
        double devBigramPerplexity = evaluateSet(devNewFilePath,2);
        System.out.println(devBigramPerplexity);
        double testBigramPerplexity = evaluateSet(testNewFilePath,2);
        System.out.println(testBigramPerplexity);

        //run on train,dev,test and give prob of each sentence
        System.out.println("TRIGRAM PROBABILITIES");
        double trainTrigramPerplexity = evaluateSet(trainNewFilePath,3);
        System.out.println(trainTrigramPerplexity);
        double devTrigramPerplexity = evaluateSet(devNewFilePath,3);
        System.out.println(devTrigramPerplexity);
        double testTrigramPerplexity = evaluateSet(testNewFilePath,3);
        System.out.println(testTrigramPerplexity);

    }

    public static double evaluateSet(String filePath, int n) {

        int sum = 0;

            for (int f : unigram_map.values()) {
                sum += f;
            }
        

        double l = 0;
        try {
            /* Open the file */
            FileInputStream fstream = new FileInputStream(filePath);

            BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

            String strLine;

            int m = 0;
            double totalLogProb = 0;

            //Read File Line By Line
            while ((strLine = br.readLine()) != null) {

                ArrayList<String> tokens = new ArrayList<String>(Arrays.asList(strLine.split(" ")));
                m+=tokens.size();


                for (int i = 1; i < n; i++) {
                    tokens.add(0, startSymbol);
                }

                tokens.add(stopSymbol);

                //unigram state
                if(n==1)
                {
                   double logProb = 0;
                   //double prob = 1;

                    for (int i=0;i<tokens.size();i++) {

                        String token = tokens.get(i);
                        int tokenValue = unigram_map.get(token);

                        //System.out.println(" tokenValue: "+tokenValue);

                        //prob *= (tokenValue/sum);
                        logProb += (Math.log(tokenValue)-Math.log(sum));

                    }
                    //System.out.println(strLine+" prob: "+prob);
                    totalLogProb+=logProb;

                }

                if(n==2)
                {
                    double logProb = 0;
                    boolean foundZero = false;
                    //double prob = 1;

                    for (int i=1;i<tokens.size();i++) {

                        String token = tokens.get(i);
                        String previouskeyString = tokens.get(i-1);
                        String keyString = previouskeyString+keySeparator+token;
                        int tokenValue  = bigram_map.containsKey(keyString) ? bigram_map.get(keyString): 0;


                        //System.out.println(" tokenValue: "+tokenValue);

                        //prob *= (tokenValue/sum);
                        int previousValue = 0;
                        if(previouskeyString.equals(startSymbol))
                        {
                            previousValue = numLinesInTrain;
                        }
                        else
                        {
                            previousValue = unigram_map.containsKey(previouskeyString)?unigram_map.get(previouskeyString):0;
                        }

                        if(tokenValue == 0 || previousValue == 0)
                        {
                            foundZero = true;
                            logProb = 0;
                        }
                        else {
                            logProb += (Math.log(tokenValue) - Math.log(previousValue));
                        }

                    }
                    //System.out.println(strLine+" prob: "+prob);
                    if(foundZero)
                        logProb = 0;
                    totalLogProb+=logProb;

                }

                if(n==3)
                {
                    double logProb = 0;

                    //double prob = 1;

                    for (int i=2;i<tokens.size();i++) {


                        String token = tokens.get(i);
                        String previous = tokens.get(i-1);
                        String twoPrevious = tokens.get(i-2);
                        double tokenValue = 0;
                        if(i==2)
                        {
                            //unigram first word
                            tokenValue = unigram_map.get(token);
                            logProb += (Math.log(tokenValue)-Math.log(sum));

                        }
                        if(i==3)
                        {
                            //bigram second word
                            String keyString = String.join(keySeparator,new String[] {previous,token});
                            tokenValue = bigram_map.containsKey(keyString)?bigram_map.get(keyString):Double.MIN_VALUE;
                            logProb += (Math.log(tokenValue)-Math.log(unigram_map.get(previous)));
                        }
                        else
                        {
                            String keyString = String.join(keySeparator,new String[] {twoPrevious,previous,token});
                            String previouskeyString = String.join(keySeparator,new String[] {twoPrevious,previous});
                            tokenValue = trigram_map.containsKey(keyString)?trigram_map.get(keyString):Double.MIN_VALUE;
                            double previousValue = bigram_map.containsKey(previouskeyString)?bigram_map.get(previouskeyString):Double.MIN_VALUE;
                            logProb += (Math.log(tokenValue)-Math.log(previousValue));
                        }



                        //System.out.println(" tokenValue: "+tokenValue);
                      /*  double previousValue = 0;
                        if(previouskeyString.equals(String.join(keySeparator,new String[]{startSymbol,startSymbol})))
                        {
                            previousValue = numLinesInTrain;
                        }
                        else
                        {
                            previousValue = bigram_map.containsKey(previouskeyString)?bigram_map.get(previouskeyString):Double.MIN_VALUE;
                        }

                            logProb += (Math.log(tokenValue)-Math.log(previousValue));*/

                        //prob *= (tokenValue/sum);

                    }
                    //System.out.println(strLine+" prob: "+prob);


                    totalLogProb+=logProb;

                }
            }
            System.out.println("total log prob: "+totalLogProb);
            l = (totalLogProb/m);
            //Close the input stream
            br.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

        return Math.pow(2,-1*l);
    }

    public static void remove_low_freq_words(String inputPath, String outputPath, HashMap<String,Integer> words) {

        try {
            /* Open the file */
            FileInputStream fstream = new FileInputStream(inputPath);

            BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

            BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath));


            String strLine;

            //Read File Line By Line
            while ((strLine = br.readLine()) != null) {

                ArrayList<String> list = new ArrayList();

                ArrayList<String> tokens = new ArrayList<String>(Arrays.asList(strLine.split(" ")));

                for (int i = 0; i < tokens.size(); i++) {
                    String keyString = tokens.get(i);
                    if(!words.containsKey(keyString))
                    {
                        keyString = unkSymbol;
                        tokens.set(i,keyString);
                    }
                }

                String newLine = String.join(" ",tokens);

                writer.write(newLine);
                writer.newLine();
            }
            //Close the input stream
            br.close();
            writer.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static HashMap<String, Integer> token_count(String path) {
        HashMap<String, Integer> counts = new HashMap<String, Integer>();


        try {
            /* Open the file */
            FileInputStream fstream = new FileInputStream(path);

            BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

            String strLine;

            //Read File Line By Line
            while ((strLine = br.readLine()) != null) {

                ArrayList<String> list = new ArrayList();

                ArrayList<String> tokens = new ArrayList<String>(Arrays.asList(strLine.split(" ")));

                for (int i = 0; i < tokens.size(); i++) {
                    String keyString = tokens.get(i);
                    int valueCount = counts.containsKey(keyString) ? counts.get(keyString) : 0;
                    counts.put(keyString, valueCount + 1);
                }
            }
            //Close the input stream
            br.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

     /*   for (String name: counts.keySet()){

            String key =name;
            String value = counts.get(name).toString();
            System.out.println(key + " " + value);


        }*/
        return counts;
    }

    public static HashMap<String, Integer> ngram_count(String path, int n) {
        HashMap<String, Integer> counts = new HashMap<String, Integer>();


        try {
            /* Open the file */
            FileInputStream fstream = new FileInputStream(path);

            BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

            String strLine;

            int numLines = 0;

            //Read File Line By Line
            while ((strLine = br.readLine()) != null) {

                ArrayList<String> list = new ArrayList();
                numLines+=1;

                ArrayList<String> tokens = new ArrayList<String>(Arrays.asList(strLine.split(" ")));

                for (int i = 1; i < n; i++) {
                    tokens.add(0, startSymbol);
                }

                tokens.add(stopSymbol);

                for (int i = 0; i < tokens.size(); i++) {
                    ArrayList<String> keyArr = new ArrayList<>();
                    for (int j = i; (j < i + n) && (j< tokens.size()); j++) {

                        keyArr.add(tokens.get(j));
                    }

                    String keyString = String.join(keySeparator, keyArr);

                    int valueCount = counts.containsKey(keyString) ? counts.get(keyString) : 0;
                    counts.put(keyString, valueCount + 1);
                }

                // Print the content on the console
               // System.out.println(strLine);
            }


            //Close the input stream
            br.close();
            System.out.println(counts);
            numLinesInTrain = numLines;

        } catch (Exception e) {
            e.printStackTrace();
        }

      /*  for (String name : counts.keySet()) {

            String key = name;
            String value = counts.get(name).toString();
            System.out.println(key + " " + value);


        }*/
        return counts;
    }
}
