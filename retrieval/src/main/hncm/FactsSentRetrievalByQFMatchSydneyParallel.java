package main.hncm;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import com.FileUtil;

/**Retrieve the most relevant K (K=50) fact sentences given a query context (Context-Fact Match)
 * Note that facts for UDC come from ***sentence retrieval*** from AskUbuntu data. For each context, 
 * retrieve top 50 most relevant sentences from the QA(could be Q or A) posts in AskUbuntu data dump
 * Then remove those duplicated sentences in these top 100 retrieved sentences
 * Thus the index should also be for sentences in QA posts of AskUbuntu data dump
 * This will be the facts retrieval module in the hybrid neural conversation model
 * The data format of the UDC facts/tips file is
 * id \t context \t fact_1 \t fact_2 \t … \t fact_n \t response
 * 
 * Facts retrieval mostly done. But the speed is a bit slow. The next step is to split the 1M queries 
 * in training data of UDC into 10 pieces. Then submit 10 jobs on sydney2 to run these 1M queries over
 * the same index. Finally merge the retrieved facts into a single file. This parallel computing process
 * can reduce the time from 33 hours to 3.3 hours (X10 speed up)
 * Created in Jan. 2019
 * 
 * @author Liu Yang
 * @email  lyang@cs.umass.edu
 * @homepage https://sites.google.com/site/lyangwww/
 */

public class FactsSentRetrievalByQFMatchSydneyParallel {

public static void main(String[] args) throws IOException, ParseException{
		//java -jar factsSentRetrievalByQFMatch.jar 50 sentText udc
		if(args.length < 3){
		    	System.out.println("please input params: hitsPerPage(50) searchFieldInPosts(sentText) inputQueryFile(train_cr_pair_fl_part_aa)");
		    	System.exit(1);
	    }
		String dataPart = "train";
		int hitsPerPage = Integer.valueOf(args[0]);
	    String searchFieldInPosts = args[1], inputQueryFile = args[2]; 
	    System.out.println("input params hitsPerPage searchFieldName inputQueryFile: " + hitsPerPage + "\t" + searchFieldInPosts + "\t" + inputQueryFile);
		//String modelInputDataPath = "/home/lyang/work1Lyang/EclipseWorkspace/NLPIRConversationalQA/data/ComputeTrainFactDistributed/";
		String indexPath = "/home/lyang/work1Lyang/EclipseWorkspace/NLPIRConversationalQA/data/ExternalCollection/askubuntu/askubuntu-index-qandasents-whole/";

		Analyzer analyzer = new StandardAnalyzer();
		Directory indexDir = FSDirectory.open(Paths.get(indexPath));
	    IndexReader indexReader = DirectoryReader.open(indexDir);
	    IndexSearcher searcher = new IndexSearcher(indexReader);
	    searcher.setSimilarity(new BM25Similarity());
	    DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
		//String inputQueryFile = modelInputDataPath + dataPart + "_context_response_filter_long_seq.txt"; //context \t response
		String factsFile = inputQueryFile + ".fact"; // retrieved facts
		//The data format of the UDC facts/tips file is
		//id \t context \t fact_1 \t fact_2 \t … \t fact_n \t response
		System.out.println("generate factsFile: " + factsFile);
		FileWriter factsWriter = new FileWriter(factsFile);
		ArrayList<String> inputLines = new ArrayList<>();
		FileUtil.readLines(inputQueryFile, inputLines);
		int instanceID = 0;
		for(String line : inputLines){
			//System.out.println("test line: " + line);
			//context \t response
			String context = line.split("\t")[0], response = line.split("\t")[1], query = context.trim();
			//System.out.println("test tokens: " + tokens);
//				for(String s : tokens) {
//					System.out.print(s + " ");
//				}
//				System.out.println();
			//label query candidate_response
			doRetrieval(query, context, response, analyzer, searcher,hitsPerPage, factsWriter, instanceID, searchFieldInPosts, dataPart);
			instanceID++;
			if(instanceID % 500 == 0){
				System.out.println("cur dataPart and query/instance id: " + dataPart + "\t" + instanceID);
				LocalDateTime now = LocalDateTime.now();
				System.out.println("current time: " + dtf.format(now));
			}
		}
		factsWriter.close();
		
	}

	//Given a test/dev/train context query, retrieve facts from QA sentence index in AskUbuntu dump
	//instanceID is the ID for the train/valid/test instance
	//add train/valid/test in instanceID to discriminate instances in train/valid/test files
	private static void doRetrieval(String query, String context, String response, Analyzer analyzer, IndexSearcher searcher,
			int hitsPerPage, FileWriter factsWriter, int instanceID, String searchFieldInPosts, String dataPart) throws ParseException, IOException {
		// TODO Auto-generated method stub
		ArrayList<String> qWords = new ArrayList<>();
		FileUtil.tokenizeStanfordTKAndLowerCase(query, qWords, false); //tokenization, remove punctuation, lower case, remove stop words (if set to true)
		String qString = "";
		for(String qw : qWords){
			qString += " " + qw;
		}
		// System.out.println("test qString: " + qString);
		if(qString.trim().equals("") || qString.trim() == null) {
			//The data format of the UDC facts/tips file is
			//id \t context \t fact_1 \t fact_2 \t … \t fact_n \t response
			factsWriter.append(dataPart + "_" + instanceID + "\t" + context + "\tNULL\t" + response + "\n"); 
			factsWriter.flush();
			return; //If qString is null (question words are all stop words), add a line with NULL and return.
		}
		BooleanQuery.setMaxClauseCount(102400);
		Query q = new QueryParser(searchFieldInPosts, analyzer).parse(QueryParser.escape(qString));
		TopScoreDocCollector collector = TopScoreDocCollector.create(hitsPerPage);
		searcher.search(q, collector);
	    ArrayList<ScoreDoc> hitsList = new ArrayList<ScoreDoc>();
	    for(ScoreDoc s : collector.topDocs().scoreDocs){
	    		hitsList.add(s);
	    }
	    
	    //output the top retrieved documents
	    if(hitsList.size() <= 0) { // no hits. It is possible that there is no hits for this query context, then there is only one retrieved line (null) for this query
	    		factsWriter.append(dataPart + "_" + instanceID + "\t" + context + "\tNULL\t" + response + "\n"); 
			factsWriter.flush();
			return;
		}
	    String factLine = dataPart + "_" + instanceID + "\t" + context + "\t";
	    //Define a cache set to remove duplicated (exactly the same) facts in top ranked retrieved results
	    Set<String> cacheFacts = new HashSet<>();
	    for(int i = 0; i < hitsList.size(); i++){
	    		int docId = hitsList.get(i).doc;
		    Document d = searcher.doc(docId);
		    String fact = d.get(searchFieldInPosts);
		    if(cacheFacts.size() == 0 || !cacheFacts.contains(fact)) {//only add non-duplicated facts
		    		factLine += fact + "\t";
		    		cacheFacts.add(fact);
		    }
	    }
	    factLine += response + "\n";
	    cacheFacts.clear();
	    factsWriter.append(factLine);
	    factsWriter.flush();
	}
}
