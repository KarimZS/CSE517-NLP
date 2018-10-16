import java.io.*;
import java.sql.Struct;
import java.util.*;

public class Main {

    public static String startSymbol = "*START*";
    public static String stopSymbol = "*STOP*";
    public static String unkSymbol = "*UNK*";
    public static String keySeparator = "#";

    public static int unkThreshold = 2;


    public static void main(String[] args) {
        // write your code here
        String trainFilePath = "C:\\Users\\karim\\OneDrive\\Documents\\uwash\\fall18\\repo\\CSE517-NLP\\hw1\\CSE517-HW1-Data\\prob1_brown_full\\brown.train.txt";
        String devFilePath = "C:\\Users\\karim\\OneDrive\\Documents\\uwash\\fall18\\repo\\CSE517-NLP\\hw1\\CSE517-HW1-Data\\prob1_brown_full\\brown.dev.txt";
        String testFilePath = "C:\\Users\\karim\\OneDrive\\Documents\\uwash\\fall18\\repo\\CSE517-NLP\\hw1\\CSE517-HW1-Data\\prob1_brown_full\\brown.test.txt";


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
        String newFilePath = "C:\\Users\\karim\\OneDrive\\Documents\\uwash\\fall18\\repo\\CSE517-NLP\\hw1\\CSE517-HW1-Data\\prob1_brown_full\\brown.train.filtered.txt";

        //go through file and replace all less than 2 words with UNK
        remove_low_freq_words(trainFilePath,newFilePath,wordsFiltered);

        //create 1,2,3 grams
        HashMap<String, Integer> unigram_map = ngram_count(newFilePath, 1);
       // HashMap<String, Integer> bigram_map = ngram_count(newFilePath, 2);
       // HashMap<String, Integer> trigram_map = ngram_count(newFilePath, 3);

        //run on train,dev,test and give prob of each sentence
        double trainUnigramPerplexity = evaluateSet(unigram_map,trainFilePath,1);
        System.out.println(trainUnigramPerplexity);
       /* double devUnigramPerplexity = evaluateSet(unigram_map,trainFilePath,1);
        double testUnigramPerplexity = evaluateSet(unigram_map,trainFilePath,1);

        double trainBigramPerplexity = evaluateSet(unigram_map,trainFilePath,2);
        double devBigramPerplexity = evaluateSet(unigram_map,trainFilePath,2);
        double testBigramPerplexity = evaluateSet(unigram_map,trainFilePath,2);

        double trainTrigramPerplexity = evaluateSet(unigram_map,trainFilePath,3);
        double devTrigramPerplexity = evaluateSet(unigram_map,trainFilePath,3);
        double testTrigramPerplexity = evaluateSet(unigram_map,trainFilePath,3);
*/
    }

    public static double evaluateSet(HashMap<String, Integer> countMap, String filePath, int n) {

        int sum = 0;
        if(n==1) {
            for (int f : countMap.values()) {
                sum += f;
            }
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

                    for (String token :tokens) {

                        int tokenValue = 0;
                        if(countMap.containsKey(token))
                        {
                            tokenValue = countMap.get(token);
                        }
                        else {
                            tokenValue = countMap.get(unkSymbol);
                        }
                        //System.out.println(" tokenValue: "+tokenValue);

                        //prob *= (tokenValue/sum);
                        logProb += (Math.log(tokenValue)-Math.log(sum));

                    }
                    //System.out.println(strLine+" prob: "+prob);
                    totalLogProb+=logProb;

                }

             /*   for (int i = 0; i < tokens.size(); i++) {
                    ArrayList<String> keyArr = new ArrayList<>();
                    for (int j = i; j < i + n; j++) {
                        if (j >= tokens.size()) {
                            break;
                        }
                        keyArr.add(tokens.get(j));
                    }

                    String keyString = String.join(keySeparator, keyArr);
                    int KeyStringCount = 0;
                    if(countMap.containsKey(keyString))
                    {
                        KeyStringCount = countMap.get(keyString);
                    }
                    else {
                        KeyStringCount = countMap.get(unkSymbol);
                    }

                    int valueCount = counts.containsKey(keyString) ? counts.get(keyString) : 0;
                    counts.put(keyString, valueCount + 1);
                }

                // Print the content on the console
                System.out.println(strLine);*/
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

            //Read File Line By Line
            while ((strLine = br.readLine()) != null) {

                ArrayList<String> list = new ArrayList();

                ArrayList<String> tokens = new ArrayList<String>(Arrays.asList(strLine.split(" ")));

                for (int i = 1; i < n; i++) {
                    tokens.add(0, startSymbol);
                }

                tokens.add(stopSymbol);

                for (int i = 0; i < tokens.size(); i++) {
                    ArrayList<String> keyArr = new ArrayList<>();
                    for (int j = i; j < i + n; j++) {
                        if (j >= tokens.size()) {
                            break;
                        }
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
