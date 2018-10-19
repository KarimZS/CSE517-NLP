import com.sun.tools.corba.se.idl.InterfaceGen;

import java.io.*;
import java.sql.Struct;
import java.util.*;

public class Main {

    public static String startSymbol = "START";
    public static String stopSymbol = "STOP";
    public static String unkSymbol = "UNK";
    public static String keySeparator = "#";

    public static int unkThreshold = 5;
    public static int addK = 1;
    public static double unigramLambda = .2;
    public static double bigramLambda = .2;
    public static double trigramLambda = .6;




    public static HashMap<String, Integer> unigram_map = null;
    public static HashMap<String, HashMap<String, Integer>> bigram_map = null;
    public static HashMap<String, HashMap<String, Integer>> trigram_map = null;
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
        bigram_map = bigram_count(trainNewFilePath);
        trigram_map =  trigram_count(trainNewFilePath);

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

        System.out.println("BIGRAM PROBABILITIES-Add K Smoothed");
        double addKtrainBigramPerplexity = addKevaluateSet(trainNewFilePath,2,addK);
        System.out.println(addKtrainBigramPerplexity);
        double addKdevBigramPerplexity = addKevaluateSet(devNewFilePath,2,addK);
        System.out.println(addKdevBigramPerplexity);
        double addKtestBigramPerplexity = addKevaluateSet(testNewFilePath,2,addK);
        System.out.println(addKtestBigramPerplexity);

        //run on train,dev,test and give prob of each sentence
        System.out.println("TRIGRAM PROBABILITIES");
        double trainTrigramPerplexity = evaluateSet(trainNewFilePath,3);
        System.out.println(trainTrigramPerplexity);
        double devTrigramPerplexity = evaluateSet(devNewFilePath,3);
        System.out.println(devTrigramPerplexity);
        double testTrigramPerplexity = evaluateSet(testNewFilePath,3);
        System.out.println(testTrigramPerplexity);

        System.out.println("TRIGRAM PROBABILITIES-Add K Smoothed");
        double addKtrainTrigramPerplexity = addKevaluateSet(trainNewFilePath,3,addK);
        System.out.println(addKtrainTrigramPerplexity);
        double addKdevTrigramPerplexity = addKevaluateSet(devNewFilePath,3,addK);
        System.out.println(addKdevTrigramPerplexity);
        double addKtestTrigramPerplexity = addKevaluateSet(testNewFilePath,3,addK);
        System.out.println(addKtestTrigramPerplexity);

        System.out.println("TRIGRAM PROBABILITIES-Linear Interpolation Smoothed");
        double lininttrainTrigramPerplexity = linearInterpolationevaluateSet(trainNewFilePath,3,unigramLambda,bigramLambda,trigramLambda);
        System.out.println(lininttrainTrigramPerplexity);
        double linintdevTrigramPerplexity = linearInterpolationevaluateSet(devNewFilePath,3,unigramLambda,bigramLambda,trigramLambda);
        System.out.println(linintdevTrigramPerplexity);
        double lininttestTrigramPerplexity = linearInterpolationevaluateSet(testNewFilePath,3,unigramLambda,bigramLambda,trigramLambda);
        System.out.println(lininttestTrigramPerplexity);

    }
    public static double linearInterpolationevaluateSet(String filePath, int n,double l1,double l2,double l3) {

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

                tokens.add(stopSymbol);

                //unigram state
                if(n==1)
                {
                    double logProb = 0;
                    //double prob = 1;

                    for (int i=0;i<tokens.size();i++) {

                        String token = tokens.get(i);
                        //unigram first word
                        double tokenValue = unigram_map.get(token);
                        logProb += (Math.log(l1*tokenValue)-Math.log(sum));
                    }
                    //System.out.println(strLine+" prob: "+prob);
                    totalLogProb+=logProb;

                }

                if(n==2)
                {
                    double logProb = 0;

                    for (int i=0;i<tokens.size();i++) {

                        if(i==0)
                        {
                            String token = tokens.get(i);
                            //unigram first word
                            double tokenValue = unigram_map.get(token);
                            logProb += (Math.log(l1*tokenValue)-Math.log(sum));

                        }
                        else
                        {

                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);

                            double tokenValue = 0;
                            double denominator = unigram_map.get(previous);

                            HashMap<String,Integer> bigramTokenTable = bigram_map.containsKey(previous)?bigram_map.get(previous):null;
                            if(bigramTokenTable!=null)
                            {
                                tokenValue = bigramTokenTable.containsKey(token)?bigramTokenTable.get(token):0;

                            }
                            else
                            {
                                tokenValue = 0;

                            }

                            double unigramTokenValue = unigram_map.get(token);

                            if(tokenValue!=0)
                            {
                                logProb += (Math.log(l1*unigramTokenValue)-Math.log(sum));
                                logProb += (Math.log(l2*tokenValue)-Math.log(denominator));
                            }
                            else
                            {
                                logProb += (Math.log(l1*unigramTokenValue)-Math.log(sum));
                            }
                        }
                    }
                    //System.out.println(strLine+" prob: "+prob);
                    totalLogProb+=logProb;

                }

                if(n==3)
                {

                    double logProb = 0;

                    for (int i=0;i<tokens.size();i++) {

                        if(i==0)
                        {
                            String token = tokens.get(i);
                            //unigram first word
                            double tokenValue = unigram_map.get(token);
                            logProb += (Math.log(l1*tokenValue)-Math.log(sum));

                        }
                        else if(i==1)
                        {
                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);

                            double tokenValue = 0;
                            double denominator = unigram_map.get(previous);

                            HashMap<String,Integer> bigramTokenTable = bigram_map.containsKey(previous)?bigram_map.get(previous):null;
                            if(bigramTokenTable!=null)
                            {
                                tokenValue = bigramTokenTable.containsKey(token)?bigramTokenTable.get(token):0;

                            }
                            else
                            {
                                tokenValue = 0;

                            }

                            double unigramTokenValue = unigram_map.get(token);

                            if(tokenValue!=0)
                            {
                                logProb += (Math.log(l1*unigramTokenValue)-Math.log(sum));
                                logProb += (Math.log(l2*tokenValue)-Math.log(denominator));
                            }
                            else
                            {
                                logProb += (Math.log(l1*unigramTokenValue)-Math.log(sum));
                            }
                        }
                        else
                        {
                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);
                            String twoPrevious = tokens.get(i-2);
                            String previouskeyString = String.join(keySeparator,new String[] {twoPrevious,previous});


                            double tokenValue = 0;
                            double denominator =bigram_map.containsKey(twoPrevious)?bigram_map.get(twoPrevious).containsKey(previous)?bigram_map.get(twoPrevious).get(previous):0:0;

                            HashMap<String,Integer> temp = trigram_map.containsKey(previouskeyString)?trigram_map.get(previouskeyString):null;
                            if(temp!=null)
                            {
                                tokenValue = temp.containsKey(token)?temp.get(token):0;

                            }
                            else
                            {
                                tokenValue = 0;
                            }

                            double unigramTokenValue = unigram_map.get(token);

                            //bigram token value
                            double bigramtokenValue = 0;
                            double bigramdenominator = unigram_map.get(previous);

                            HashMap<String,Integer> bigramTokenTable = bigram_map.containsKey(previous)?bigram_map.get(previous):null;
                            if(bigramTokenTable!=null)
                            {
                                bigramtokenValue = bigramTokenTable.containsKey(token)?bigramTokenTable.get(token):0;

                            }
                            else
                            {
                                bigramtokenValue = 0;

                            }

                            if(tokenValue!=0&&bigramtokenValue!=0)
                            {
                                logProb += (Math.log(l1*unigramTokenValue)-Math.log(sum));
                                logProb += (Math.log(l2*bigramtokenValue)-Math.log(bigramdenominator));
                                logProb += (Math.log(l3*tokenValue)-Math.log(denominator));
                            }
                            else if(bigramtokenValue!=0)
                            {
                                logProb += (Math.log(l1*unigramTokenValue)-Math.log(sum));
                                logProb += (Math.log(l2*bigramtokenValue)-Math.log(bigramdenominator));
                            }
                            else
                            {
                                logProb += (Math.log(l1*unigramTokenValue)-Math.log(sum));
                            }
                        }
                    }
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

    public static double addKevaluateSet(String filePath, int n,int k) {

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

                tokens.add(stopSymbol);

                //unigram state
                if(n==1)
                {
                    double logProb = 0;
                    //double prob = 1;

                    for (int i=0;i<tokens.size();i++) {

                        String token = tokens.get(i);
                        //unigram first word
                        double tokenValue = unigram_map.get(token);
                        logProb += (Math.log(tokenValue)-Math.log(sum));
                    }
                    //System.out.println(strLine+" prob: "+prob);
                    totalLogProb+=logProb;

                }

                if(n==2)
                {
                    double logProb = 0;

                    for (int i=0;i<tokens.size();i++) {

                        if(i==0)
                        {
                            String token = tokens.get(i);
                            //unigram first word
                            double tokenValue = unigram_map.get(token);
                            logProb += (Math.log(tokenValue)-Math.log(sum));

                        }
                        else
                        {

                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);

                            double tokenValue = 0;
                            double denominator = 0;

                            HashMap<String,Integer> bigramTokenTable = bigram_map.containsKey(previous)?bigram_map.get(previous):null;

                            tokenValue = k+ (bigramTokenTable.containsKey(token)?bigramTokenTable.get(token):0);
                            denominator = unigram_map.get(previous) + k*(unigram_map.keySet().size());

                            logProb += (Math.log(tokenValue+k)-Math.log(denominator));
                        }
                    }
                    //System.out.println(strLine+" prob: "+prob);
                    totalLogProb+=logProb;

                }

                if(n==3)
                {
                    double logProb = 0;

                    for (int i=0;i<tokens.size();i++) {

                        if(i==0)
                        {
                            String token = tokens.get(i);
                            //unigram first word
                            double tokenValue = unigram_map.get(token);
                            logProb += (Math.log(tokenValue)-Math.log(sum));

                        }
                        else if(i==1)
                        {
                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);

                            double tokenValue = 0;
                            double denominator = 0;

                            HashMap<String,Integer> bigramTokenTable = bigram_map.containsKey(previous)?bigram_map.get(previous):null;

                            tokenValue = k+ (bigramTokenTable.containsKey(token)?bigramTokenTable.get(token):0);
                            denominator = unigram_map.get(previous) + k*(unigram_map.keySet().size());

                            logProb += (Math.log(tokenValue+k)-Math.log(denominator));
                        }
                        else
                        {
                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);
                            String twoPrevious = tokens.get(i-2);
                            String previouskeyString = String.join(keySeparator,new String[] {twoPrevious,previous});


                            double tokenValue = 0;
                            double denominator = 0;

                            HashMap<String,Integer> trigramTokenTable = trigram_map.containsKey(previouskeyString)?trigram_map.get(previouskeyString):null;
                            if(trigramTokenTable!=null)
                            {
                                tokenValue = k+ (trigramTokenTable.containsKey(token)?trigramTokenTable.get(token):0);
                                denominator = k*unigram_map.keySet().size() + (bigram_map.containsKey(twoPrevious)?bigram_map.get(twoPrevious).containsKey(previous)?bigram_map.get(twoPrevious).get(previous):0:0);
                            }
                            else
                            {
                                tokenValue = k;
                                denominator =k*unigram_map.keySet().size();
                            }
                            logProb += (Math.log(tokenValue+k)-Math.log(denominator));
                        }

                    }

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

                tokens.add(stopSymbol);

                //unigram state
                if(n==1)
                {
                   double logProb = 0;
                   //double prob = 1;

                    for (int i=0;i<tokens.size();i++) {

                        String token = tokens.get(i);
                        //unigram first word
                        double tokenValue = unigram_map.get(token);
                        logProb += (Math.log(tokenValue)-Math.log(sum));
                    }
                    //System.out.println(strLine+" prob: "+prob);
                    totalLogProb+=logProb;

                }

                if(n==2)
                {
                    double logProb = 0;

                    for (int i=0;i<tokens.size();i++) {

                        if(i==0)
                        {
                            String token = tokens.get(i);
                            //unigram first word
                            double tokenValue = unigram_map.get(token);
                            logProb += (Math.log(tokenValue)-Math.log(sum));

                        }
                        else
                        {

                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);

                            double tokenValue = 0;
                            double denominator = unigram_map.get(previous);

                            HashMap<String,Integer> bigramTokenTable = bigram_map.containsKey(previous)?bigram_map.get(previous):null;
                            if(bigramTokenTable!=null)
                            {
                                tokenValue = bigramTokenTable.containsKey(token)?bigramTokenTable.get(token):0;

                            }
                            else
                            {
                                tokenValue = 0;

                            }
                            logProb += (Math.log(tokenValue)-Math.log(denominator));
                        }
                    }
                    //System.out.println(strLine+" prob: "+prob);
                    totalLogProb+=logProb;

                }

                if(n==3)
                {

                    double logProb = 0;

                    for (int i=0;i<tokens.size();i++) {

                        if(i==0)
                        {
                            String token = tokens.get(i);
                            //unigram first word
                            double tokenValue = unigram_map.get(token);
                            logProb += (Math.log(tokenValue)-Math.log(sum));

                        }
                        else if(i==1)
                        {
                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);

                            double tokenValue = 0;
                            double denominator = unigram_map.get(previous);

                            HashMap<String,Integer> bigramTokenTable = bigram_map.containsKey(previous)?bigram_map.get(previous):null;
                            if(bigramTokenTable!=null)
                            {
                                tokenValue = bigramTokenTable.containsKey(token)?bigramTokenTable.get(token):0;

                            }
                            else
                            {
                                tokenValue = 0;

                            }
                            logProb += (Math.log(tokenValue)-Math.log(denominator));
                        }
                        else
                        {
                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);
                            String twoPrevious = tokens.get(i-2);
                            String previouskeyString = String.join(keySeparator,new String[] {twoPrevious,previous});


                            double tokenValue = 0;
                            double denominator =bigram_map.containsKey(twoPrevious)?bigram_map.get(twoPrevious).containsKey(previous)?bigram_map.get(twoPrevious).get(previous):0:0;

                            HashMap<String,Integer> temp = trigram_map.containsKey(previouskeyString)?trigram_map.get(previouskeyString):null;
                            if(temp!=null)
                            {
                                tokenValue = temp.containsKey(token)?temp.get(token):0;

                            }
                            else
                            {
                                 tokenValue = 0;
                            }
                            logProb += (Math.log(tokenValue)-Math.log(denominator));
                        }

                    }

                    totalLogProb+=logProb;
                    /*double logProb = 0;

                    for (int i=0;i<tokens.size();i++) {

                        if(i==0)
                        {
                            String token = tokens.get(i);
                            //unigram first word
                            double tokenValue = unigram_map.get(token);
                            logProb += (Math.log(tokenValue)-Math.log(sum));

                        }
                        else if(i==1)
                        {
                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);
                            //bigram second word
                            String keyString = String.join(keySeparator,new String[] {previous,token});
                            double tokenValue = bigram_map.containsKey(keyString)?bigram_map.get(keyString):0;
                            logProb += (Math.log(tokenValue)-Math.log(unigram_map.get(previous)));
                        }
                        else
                        {
                            String token = tokens.get(i);
                            String previous = tokens.get(i-1);
                            String twoPrevious = tokens.get(i-2);
                            String keyString = String.join(keySeparator,new String[] {twoPrevious,previous,token});
                            String previouskeyString = String.join(keySeparator,new String[] {twoPrevious,previous});
                            double tokenValue = trigram_map.containsKey(keyString)?trigram_map.get(keyString):0;
                            double previousValue = bigram_map.containsKey(previouskeyString)?bigram_map.get(previouskeyString):0;
                            logProb += (Math.log(tokenValue)-Math.log(previousValue));
                        }

                    }

                    totalLogProb+=logProb;*/
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

                //  for (int i = 1; i < n; i++) {
                //      tokens.add(0, startSymbol);
                //  }

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
           //System.out.println(counts);
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

    public static HashMap<String, HashMap<String,Integer>> bigram_count(String path) {
        HashMap<String, Integer> counts = new HashMap<String, Integer>();
        HashMap<String, HashMap<String,Integer>> res = new HashMap<String, HashMap<String,Integer>>();

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

                //  for (int i = 1; i < n; i++) {
                //      tokens.add(0, startSymbol);
                //  }

                tokens.add(stopSymbol);

                for (int i = 1; i < tokens.size(); i++) {
                    String previous = tokens.get(i-1);
                    String token = tokens.get(i);
                    int value =0;

                    //String keyString = String.join(keySeparator, new String[] {twoPrevious,previous});
                    if(res.containsKey(previous))
                    {
                        HashMap<String, Integer> temp = res.get(previous);
                        if(temp.containsKey(token))
                        {
                            value = temp.get(token);
                        }
                        else
                        {
                            value = 0;
                        }
                        temp.put(token,value+1);
                        res.put(previous,temp);
                    }
                    else
                    {
                        HashMap<String, Integer> temp = new HashMap<String,Integer>();
                        temp.put(token,1);
                        res.put(previous,temp);
                    }
                }

                // Print the content on the console
                // System.out.println(strLine);
            }


            //Close the input stream
            br.close();
            //System.out.println(counts);
            numLinesInTrain = numLines;

        } catch (Exception e) {
            e.printStackTrace();
        }

      /*  for (String name : counts.keySet()) {

            String key = name;
            String value = counts.get(name).toString();
            System.out.println(key + " " + value);


        }*/
        return res;
    }

    public static HashMap<String, HashMap<String,Integer>> trigram_count(String path) {
        HashMap<String, Integer> counts = new HashMap<String, Integer>();
        HashMap<String, HashMap<String,Integer>> res = new HashMap<String, HashMap<String,Integer>>();

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

                //  for (int i = 1; i < n; i++) {
                //      tokens.add(0, startSymbol);
                //  }

                tokens.add(stopSymbol);

                for (int i = 2; i < tokens.size(); i++) {
                    String previous = tokens.get(i-1);
                    String twoPrevious = tokens.get(i-2);
                    String token = tokens.get(i);
                    int value =0;

                    String keyString = String.join(keySeparator, new String[] {twoPrevious,previous});
                    if(res.containsKey(keyString))
                    {
                        HashMap<String, Integer> temp = res.get(keyString);
                        if(temp.containsKey(token))
                        {
                            value = temp.get(token);
                        }
                        else
                        {
                           value = 0;
                        }
                        temp.put(token,value+1);
                        res.put(keyString,temp);
                    }
                    else
                    {
                        HashMap<String, Integer> temp = new HashMap<String,Integer>();
                        temp.put(token,1);
                        res.put(keyString,temp);
                    }
                }

                // Print the content on the console
                // System.out.println(strLine);
            }


            //Close the input stream
            br.close();
            //System.out.println(counts);
            numLinesInTrain = numLines;

        } catch (Exception e) {
            e.printStackTrace();
        }

      /*  for (String name : counts.keySet()) {

            String key = name;
            String value = counts.get(name).toString();
            System.out.println(key + " " + value);


        }*/
        return res;
    }
}
