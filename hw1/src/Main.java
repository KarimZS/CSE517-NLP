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
        String filePath = "C:\\Users\\karim\\OneDrive\\Documents\\uwash\\fall18\\repo\\CSE517-NLP\\hw1\\CSE517-HW1-Data\\prob1_brown_full\\brown.train.txt";

        //get words with count less than 2
        HashMap<String, Integer> words = token_count(filePath);
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
        remove_low_freq_words(filePath,newFilePath,wordsFiltered);

        //create 1,2,3 grams
        HashMap<String, Integer> unigram_map = ngram_count(newFilePath, 1);
        HashMap<String, Integer> bigram_map = ngram_count(newFilePath, 2);
        HashMap<String, Integer> trigram_map = ngram_count(newFilePath, 3);

        //run on train,dev,test and give prob of each sentence


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
                System.out.println(strLine);
            }


            //Close the input stream
            br.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

        for (String name : counts.keySet()) {

            String key = name;
            String value = counts.get(name).toString();
            System.out.println(key + " " + value);


        }
        return counts;
    }
}
