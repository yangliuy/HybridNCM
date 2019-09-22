package main.hncm;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.ElementHandler;
import org.dom4j.ElementPath;
import org.dom4j.io.SAXReader;

import parser.StanfordTokenizer;

/**Build search index for the external facts data like AskUbuntu for UDC
 * Then retrieve the most relevant K (K=50) facts given a query context (Context-Fact Match)
 * Note that facts for UDC come from ***sentence retrieval*** from AskUbuntu data. For each context, 
 * retrieve top 50 most relevant sentences from the QA(could be Q or A) posts in AskUbuntu data dump
 * Thus the index should also be for sentences in QA posts of AskUbuntu data dump
 * This will be the facts retrieval module in the hybrid neural conversation model
 * Created in July, 2018 in Microsoft Research
 * Updated in Jan. 2019
 * 
 * @author Liu Yang
 * @email  lyang@cs.umass.edu
 * @homepage https://sites.google.com/site/lyangwww/
 */

public class BuildExternalFactsSentRetIndex {
	
	public static long indexedQAPostNum = 0;
	
	public static void main(String[] args) throws IOException, InterruptedException, ParseException, DocumentException {

		//1. Build index for AskUbuntu data (index sentences in QA posts including those sentences in both title and body)
		Analyzer analyzer = new StandardAnalyzer();
//	    String indexPath = "/net/home/lyang/NLPIRConversationalQA/data/ExternalCollection/askubuntu/askubuntu-index-qandasents-whole/";
//	    String indexFileName = "/net/home/lyang/NLPIRConversationalQA/data/ExternalCollection/askubuntu/Posts.xml";
	    if(args.length < 2){
	    	System.out.println("please input params: indexPath indexFileName");
	    	System.exit(1);
	    }
	    String indexPath = args[0];
	    String indexFileName = args[1];
	    System.out.println("indexPath: " + indexPath);
	    System.out.println("indexFileName: " + indexFileName);
	    int maxFactLen = 80, minFactLen = 3; // the max and min length of fact sentences
	    if(!new File(indexPath).exists()){ //If the index path is not existed, created it
		    	System.out.println("create index path: " + indexPath);
		    	new File(indexPath).mkdir();
	    }
	    Directory indexDir = FSDirectory.open(Paths.get(indexPath));
	    // Optional: for better indexing performance, if you
	    // are indexing many documents, increase the RAM
	    // buffer.  But if you do this, increase the max heap
	    // size to the JVM (eg add -Xmx512m or -Xmx1g):
	    // iwc.setRAMBufferSizeMB(1024.0);
	    System.out.println("Indexing to directory " + indexPath + "...");
	    // If there are already files under the indexPath, clear them firstly
	    for(File f : new File(indexPath).listFiles()){
	    	f.delete();
	    	System.out.println("delete file: " + f.getAbsolutePath());
	    }

	    IndexWriterConfig config = new IndexWriterConfig(analyzer);
	    //Set some parameters for faster indexing purpose
	    //Set setMaxBufferedDocs or setRAMBufferSizeMB
	    //config.setMaxBufferedDocs(Integer.MAX_VALUE);
	    config.setRAMBufferSizeMB(2048); //2GB
	    IndexWriter w = new IndexWriter(indexDir, config);
	    
	    SAXReader reader = new SAXReader();
        reader.setDefaultHandler(new ElementHandler() {
             public void onEnd(ElementPath ep) {
                  //System.out.println("see end label!");
            	  indexedQAPostNum++;
            	  if(indexedQAPostNum % 100000 == 0){
     				System.out.println("cur indexed doc number: " + indexedQAPostNum);
            	  }
                  Element e = ep.getCurrent();
                  Document curD = new Document();
                  // use a string field for isbn because we don't want it tokenized
                  String id = e.attributeValue("Id"), postTypeId = e.attributeValue("PostTypeId"), bodyText = e.attributeValue("Body"), titleText = e.attributeValue("Title");
                  String answerCount = e.attributeValue("AnswerCount");
                  // Retrieve the title of question posts to let the information more compact 
                  // and reduce nosie (Learn from the Settings of DSSM/CDSSM)
                  // Once get the relevant question, we can get the corresponding answers as the response
                  // Only need to index the title and body field of question posts
                  if(id == null || id.equals("") ||
                	 postTypeId == null || postTypeId.equals("") || // index both Qposts and Aposts
                	 bodyText == null || bodyText.equals("") ||
                	 titleText == null || titleText.equals("") || // !Filtering null title step will filter all the answer posts! So QandAPostsIndex and QpostsIndex are both QpostsIndex
                	 answerCount == null || answerCount.equals("")){
                	  	return;
                  }
                  // Maintain all questions and answers
                  //int answerCountInt = Integer.valueOf(answerCount);
                  //if(answerCountInt <= 0) return;
                  // Here we would like to do sentence retrieval
                  // Thus we use Stanford Parser to parse QA posts into sentences
                  // Then index these sentences in QA posts
                  //If there are tags, add the tokens in tags into the title
                  String tags = e.attributeValue("Tags");
                  if(tags != null && !tags.equals("")){
                	  	titleText += tags.replaceAll("[<>]", " ");
                  }
                  String allTextInPost = titleText + " " + bodyText;
                  if(allTextInPost.replaceAll(" ", "").equals("")) return;
                  ArrayList<String> sents = StanfordTokenizer.tokenizeSents(allTextInPost);
                  for(int sentID = 0; sentID < sents.size(); sentID++) {
                	  	String sent = sents.get(sentID);
                	  	if(sent == null || sent.equals("")) continue;
                	  	int sentLen = sent.split(" ").length;
                	  	if(sentLen > maxFactLen || sentLen < minFactLen) continue;//filter too long or too short fact sent
                	  	curD.add(new StringField("SentID", id + "-" + sentID, Field.Store.YES));
                    curD.add(new TextField("sentText", sent, Field.Store.YES));
                    try {
						w.addDocument(curD);
					} catch (IOException e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
					}
                  }
                  e.detach();
             }

             public void onStart(ElementPath arg0) {
                  //System.out.println("see start label!");
             }
        });
       
        org.dom4j.Document doc = reader.read(new BufferedInputStream(
                  new FileInputStream(new File(indexFileName))));
        //w.flush();
		w.close();
		System.out.println("build questions and answers sentences index done!");
	}
}