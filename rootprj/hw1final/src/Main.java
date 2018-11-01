
import java.io.*;
import java.util.*;

public class Main {

    public static String stopSymbol = "STOP";
    public static String unkSymbol = "UNK";
    public static String keySeparator = "#";

    public static int unkThreshold = 2;
    public static double addK = .01;
    public static double unigramLambda = .1;
    public static double bigramLambda = .3;
    public static double trigramLambda = .6;

    public static HashMap<String, Integer> unigram_map = null;
    public static HashMap<String, HashMap<String, Integer>> bigram_map = null;
    public static HashMap<String, HashMap<String, Integer>> trigram_map = null;
    public static int numLinesInTrain = 0;


    public static void main(String[] args) {

        String trainFilePath = "hw1final/CSE517-HW1-Data/prob1_brown_full/brown.train.txt";
        String devFilePath = "hw1final/CSE517-HW1-Data/prob1_brown_full/brown.dev.txt";
        String testFilePath = "hw1final/CSE517-HW1-Data/prob1_brown_full/brown.test.txt";


        //get words with count less than unkThreshold
        HashMap<String, Integer> words = token_count(trainFilePath);
        HashMap<String, Integer> wordsFiltered = new HashMap<>();

        Set<String> keySet = words.keySet();
        for (String key : keySet) {

            int keyValue = words.get(key);
            if (keyValue >= unkThreshold) {
                wordsFiltered.put(key,keyValue);
            }
        }
        String trainNewFilePath = "hw1final/CSE517-HW1-Data/prob1_brown_full/brown.train.filtered.txt";
        String devNewFilePath = "hw1final/CSE517-HW1-Data/prob1_brown_full/brown.dev.filtered.txt";
        String testNewFilePath = "hw1final/CSE517-HW1-Data/prob1_brown_full/brown.test.filtered.txt";

        //go through file and replace all less than unkthreshold words with UNK
        remove_low_freq_words(trainFilePath,trainNewFilePath,wordsFiltered);
        remove_low_freq_words(devFilePath,devNewFilePath,wordsFiltered);
        remove_low_freq_words(testFilePath,testNewFilePath,wordsFiltered);


        //create 1,2,3 grams
        unigram_map = ngram_count(trainNewFilePath, 1);
        bigram_map = bigram_count(trainNewFilePath);
        trigram_map =  trigram_count(trainNewFilePath);

        System.out.println();

        //run on train,dev,test and give prob of each sentence
        System.out.println("UNIGRAM PROBABILITIES");
        double trainUnigramPerplexity = evaluateSet(trainNewFilePath,1);
        System.out.println("Train "+trainUnigramPerplexity);
        double devUnigramPerplexity = evaluateSet(devNewFilePath,1);
        System.out.println("Dev "+devUnigramPerplexity);
        double testUnigramPerplexity = evaluateSet(testNewFilePath,1);
        System.out.println("Test "+testUnigramPerplexity);

        System.out.println();

        //run on train,dev,test and give prob of each sentence
        System.out.println("BIGRAM PROBABILITIES");
        double trainBigramPerplexity = evaluateSet(trainNewFilePath,2);
        System.out.println("Train "+trainBigramPerplexity);
        double devBigramPerplexity = evaluateSet(devNewFilePath,2);
        System.out.println("Dev "+devBigramPerplexity);
        double testBigramPerplexity = evaluateSet(testNewFilePath,2);
        System.out.println("Test "+testBigramPerplexity);

        System.out.println();

        //run on train,dev,test and give prob of each sentence
        System.out.println("TRIGRAM PROBABILITIES");
        double trainTrigramPerplexity = evaluateSet(trainNewFilePath,3);
        System.out.println("Train "+trainTrigramPerplexity);
        double devTrigramPerplexity = evaluateSet(devNewFilePath,3);
        System.out.println("Dev "+devTrigramPerplexity);
        double testTrigramPerplexity = evaluateSet(testNewFilePath,3);
        System.out.println("Test "+testTrigramPerplexity);

        System.out.println();

        System.out.println("TRIGRAM PROBABILITIES-Add K Smoothed");
        double addKtrainTrigramPerplexity = addKevaluateSet(trainNewFilePath,3,addK);
        System.out.println("Train "+addKtrainTrigramPerplexity);
        double addKdevTrigramPerplexity = addKevaluateSet(devNewFilePath,3,addK);
        System.out.println("Dev "+addKdevTrigramPerplexity);
        double addKtestTrigramPerplexity = addKevaluateSet(testNewFilePath,3,addK);
        System.out.println("Test "+addKtestTrigramPerplexity);

        System.out.println();

        System.out.println("TRIGRAM PROBABILITIES-Linear Interpolation Smoothed");
        double lininttrainTrigramPerplexity = linearInterpolationevaluateSet(trainNewFilePath,3,unigramLambda,bigramLambda,trigramLambda);
        System.out.println("Train "+lininttrainTrigramPerplexity);
        double linintdevTrigramPerplexity = linearInterpolationevaluateSet(devNewFilePath,3,unigramLambda,bigramLambda,trigramLambda);
        System.out.println("Dev "+linintdevTrigramPerplexity);
        double lininttestTrigramPerplexity = linearInterpolationevaluateSet(testNewFilePath,3,unigramLambda,bigramLambda,trigramLambda);
        System.out.println("Test "+lininttestTrigramPerplexity);

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

                    for (int i=0;i<tokens.size();i++) {

                        String token = tokens.get(i);
                        //unigram first word
                        double tokenValue = unigram_map.get(token);
                        logProb += (Math.log(l1*tokenValue)-Math.log(sum));
                    }
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
                            logProb += l1*(Math.log(tokenValue)-Math.log(sum));

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
                                logProb += l1*(Math.log(unigramTokenValue)-Math.log(sum));
                                logProb += l2*(Math.log(tokenValue)-Math.log(denominator));
                            }
                            else
                            {
                                logProb += l1*(Math.log(unigramTokenValue)-Math.log(sum));
                            }
                        }
                    }
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
                            logProb += l1*(Math.log(tokenValue)-Math.log(sum));
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
                                logProb += l1*(Math.log(l1*unigramTokenValue)-Math.log(sum));
                                logProb += l2*(Math.log(tokenValue)-Math.log(denominator));
                            }
                            else
                            {
                                logProb += l1*(Math.log(l1*unigramTokenValue)-Math.log(sum));
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
                                logProb += l1*(Math.log(l1*unigramTokenValue)-Math.log(sum));
                                logProb += l2*(Math.log(bigramtokenValue)-Math.log(bigramdenominator));
                                logProb += l3*(Math.log(tokenValue)-Math.log(denominator));
                            }
                            else if(bigramtokenValue!=0)
                            {
                                logProb += l1*(Math.log(l1*unigramTokenValue)-Math.log(sum));
                                logProb += l2*(Math.log(bigramtokenValue)-Math.log(bigramdenominator));
                            }
                            else
                            {
                                logProb += l1*(Math.log(unigramTokenValue)-Math.log(sum));
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

    public static double addKevaluateSet(String filePath, int n,double k) {

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

                    for (int i=0;i<tokens.size();i++) {

                        String token = tokens.get(i);
                        //unigram first word
                        double tokenValue = unigram_map.get(token);
                        logProb += (Math.log(tokenValue)-Math.log(sum));
                    }
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

                tokens.add(stopSymbol);

                for (int i = 1; i < tokens.size(); i++) {
                    String previous = tokens.get(i-1);
                    String token = tokens.get(i);
                    int value =0;

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

        return res;
    }
}
