package main.hncm;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

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

/**Retrieve the most similar K contexts given a query context (Q-Q Match)
 * Return the corresponding response of the retrieved similar context
 * This will be the retrieval module in the hybrid neural conversation model
 * Created in July, 2018 in Microsoft Research
 * 
 * @author Liu Yang
 * @email  lyang@cs.umass.edu
 * @homepage https://sites.google.com/site/lyangwww/
 */


public class OnlyRetrievalResponseByQQMatch {
	
public static void main(String[] args) throws IOException, ParseException{
		
		//String searchFieldInPosts = "bodyText"; // Can search bodyText or titleText of SO posts
		if(args.length < 3){
		    	System.out.println("please input params: hitsPerPage(15) searchFieldName(contextText) dataName(twitter or udc)");
		    	System.exit(1);
	    }
		int hitsPerPage = Integer.valueOf(args[0]);
	    String searchFieldInPosts = args[1], dataName = args[2];// the dataName could be ms or udc or twitter
	    System.out.println("input params hitsPerPage searchFieldName dataName: " + hitsPerPage + "\t" + searchFieldInPosts + "\t" + dataName);
		String modelInputDataPath = "../../PycharmProjects/NLPIRNNHybridNCMRetGen/src-hybrid-ncm/data/" + dataName +"/ModelInput/";
		String indexPath = null;
		if(dataName.equals("twitter")){
			indexPath = "data/TwitterTrainCRPairIndex/"; // on Goulburn
		} else{
			//indexPath = "data/UDCTrainCRPairIndex/"; // on Goulburn
			indexPath = "data/UDCTrainCRPairIndex/"; // index file on Goulburn 
		}

		Analyzer analyzer = new StandardAnalyzer();
		Directory indexDir = FSDirectory.open(Paths.get(indexPath));
	    IndexReader indexReader = DirectoryReader.open(indexDir);
	    IndexSearcher searcher = new IndexSearcher(indexReader);
	    searcher.setSimilarity(new BM25Similarity());
	    DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
		
		for (String dataPart : new String[]{"dev", "test", "train"}){
			String inputQueryFile = null;
			if(dataPart.equals("train")) {
				inputQueryFile = modelInputDataPath + dataPart + ".tok.clean.lc.context";
			} else {
				inputQueryFile = modelInputDataPath + dataPart + ".tok.lc.context";
			}
			
			String irResultsFile = modelInputDataPath + dataPart + "_retrieved_crpairs_seach_" + searchFieldInPosts + ".txt"; // the file storing the query and retrieval results 
			String hypFile =  modelInputDataPath + dataPart + "_retrieved_hypres_seach_" + searchFieldInPosts + ".txt"; 
			System.out.println("generate irResultsFile" + irResultsFile);
			System.out.println("generate hypFile" + hypFile);
			FileWriter irResWriter = new FileWriter(irResultsFile);
			FileWriter hypResWriter = new FileWriter(hypFile);
			ArrayList<String> inputLines = new ArrayList<>();
			FileUtil.readLines(inputQueryFile, inputLines);
			int instanceID = 0; // add train/valid/test in instanceID to discriminate instances in train/valid/test files
			for(String line : inputLines){
				//System.out.println("test line: " + line);
				String query = line.trim();
				//System.out.println("test tokens: " + tokens);
//				for(String s : tokens) {
//					System.out.print(s + " ");
//				}
//				System.out.println();
				//label query candidate_response
				doRetrieval(query, analyzer, searcher,hitsPerPage, irResWriter, hypResWriter, instanceID, searchFieldInPosts, dataPart);
				instanceID++;
				if(instanceID % 1000 == 0){
					System.out.println("cur dataPart and query/instance id: " + dataPart + "\t" + instanceID);
					LocalDateTime now = LocalDateTime.now();
					System.out.println("current time: " + dtf.format(now));
				}
			}
			irResWriter.close();
		}
	}

	//Given a test/dev/train context query, retrieve the response by BM25 Q-Q match
	//instanceID is the ID for the train/valid/test instance
	//add train/valid/test in instanceID to discriminate instances in train/valid/test files
	private static void doRetrieval(String query, Analyzer analyzer, IndexSearcher searcher,
			int hitsPerPage, FileWriter irResWriter, FileWriter hypResWriter, int instanceID, String searchFieldInPosts, String dataPart) throws ParseException, IOException {
		// TODO Auto-generated method stub
		ArrayList<String> qWords = new ArrayList<>();
		FileUtil.tokenizeStanfordTKAndLowerCase(query, qWords, false); //tokenization, remove punctuation, lower case, remove stop words (if set to true)
		String qString = "";
		for(String qw : qWords){
			qString += " " + qw;
		}
		// System.out.println("test qString: " + qString);
		if(qString.trim().equals("") || qString.trim() == null) {
			irResWriter.append(dataPart + "_" + instanceID + "\tNULL\tNULL\tNULL\n"); //  query_id \t doc_id \t retrieved_context \t retrieved_response
			hypResWriter.append("NULL\n");
			irResWriter.flush();
    		hypResWriter.flush();
			return; //If qString is null (question words are all stop words), return.
		}
		BooleanQuery.setMaxClauseCount(102400);
		Query q = new QueryParser(searchFieldInPosts, analyzer).parse(QueryParser.escape(qString));// Can search bodyText or titleText of SO posts
		TopScoreDocCollector collector = TopScoreDocCollector.create(hitsPerPage);
		searcher.search(q, collector);
	    ArrayList<ScoreDoc> hitsList = new ArrayList<ScoreDoc>();
	    for(ScoreDoc s : collector.topDocs().scoreDocs){
	    		hitsList.add(s);
	    }
	    
	    //output the top retrieved documents
	    List<String> top1Response = new ArrayList<String>();
	    if(hitsList.size() <= 0) { // no hits. It is possible that there is no hits for this query context, then there is only one retrieved line (null) for this query
			irResWriter.append(dataPart + "_" + instanceID + "\tNULL\tNULL\tNULL\n"); //  query_id \t doc_id \t retrieved_context \t retrieved_response
			hypResWriter.append("NULL\n");
			irResWriter.flush();
    		hypResWriter.flush();
			return; 
		}
	    
	    for(int i = 0; i < hitsList.size(); i++){
	    		int docId = hitsList.get(i).doc;
		    Document d = searcher.doc(docId);
		    /*if(dataPart.equals("train") && query.trim().toLowerCase().equals(d.get("contextText").trim().toLowerCase())) {
		    		//For training data, if the retrieved context is exactly the same with the query, skip it.
		    		//Prepare for distant supervision by BLEU score
		    		//System.out.println("haha! this should only be train : " + dataPart);
		    		continue;
		    }*/ // maintain the retrieved same context/response pairs to add high quality positive training examples, which is better for training a good reranker
		    irResWriter.append(dataPart + "_" + instanceID + "\t" + d.get("trainID") + 
		    		"\t" + d.get("contextText")  + "\t" + d.get("responseText") + "\n");	// query_id \t doc_id \t retrieved_context \t retrieved_response
		    irResWriter.flush();
		    if(top1Response.size() == 0) {
		    		top1Response.add(d.get("responseText"));
		    }
	    }
	    if(top1Response.size() == 1) { // only write the top-1 retrieved response into hypothesis file
	    		hypResWriter.append(top1Response.get(0) + "\n");
	    		hypResWriter.flush();
	    } else {
	    		System.out.println("Warning: can't find any top-1 retrieved response for this instance " + dataPart + "_" + instanceID);
	    		System.out.println("hitsList.size(): " + hitsList.size());
	    		hypResWriter.append("NULL\n");
	    		hypResWriter.flush();
	    }
	}
}
