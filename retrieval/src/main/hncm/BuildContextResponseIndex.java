package main.hncm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;

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

/**Build search index for the contexts in conversations such as Twitter/UDC
 * Then retrieve the most similar K contexts given a query context (Q-Q Match)
 * This will be the retrieval module (retrieve context/response pairs) in the 
 * hybrid neural conversation model
 * Created in July, 2018 in Microsoft Research
 * 
 * @author Liu Yang
 * @email  lyang@cs.umass.edu
 * @homepage https://sites.google.com/site/lyangwww/
 */

public class BuildContextResponseIndex {
	public static long indexedQAPostNum = 0;
	
	public static void main(String[] args) throws IOException, InterruptedException, ParseException, DocumentException {
		//String basedModelInputDataPath = "data/MicrosoftCSQA/v1-2/";
		
		//1. Build index for Stack Overflow & Wiki (just need to run once)
		Analyzer analyzer = new StandardAnalyzer();
		//For Twitter 
		//String indexPath = "data/TwitterTrainCRPairIndex/"; # index file on Goulburn 
		//String dataFileName = "/net/home/lyang/NLPIRNNHybridNCMRetGen/src-hybrid-ncm/data/twitter/ModelInputBackup/train_context_response.txt.mt.clean.tok.lc"; # data file on Goulburn
		//For UDC
		//String indexPath = "data/UDCTrainCRPairIndex/"; # index file on Goulburn 
		//String dataFileName = "/net/home/lyang/NLPIRNNHybridNCMRetGen/src-hybrid-ncm/data/udc/ModelInput/train.tok.clean.lc.crpairs"; # data file on Goulburn
	    
		// train.tok.clean.lc.crpairs format: context ||| response
		if(args.length < 2){
		    	System.out.println("please input params: indexPath dataFileName");
		    	System.exit(1);
	    }
	    String indexPath = args[0];
	    String indexFileName = args[1];
	    System.out.println("indexPath: " + indexPath);
	    System.out.println("indexFileName: " + indexFileName);
	    if(!new File(indexPath).exists()){ //If the index path is not existed, created it
		    	System.out.println("create index in directory : " + indexPath);
		    	new File(indexPath).mkdir();
	    }
	    Directory indexDir = FSDirectory.open(Paths.get(indexPath));
	    // Optional: for better indexing performance, if you
	    // are indexing many documents, increase the RAM
	    // buffer.  But if you do this, increase the max heap
	    // size to the JVM (eg add -Xmx512m or -Xmx1g):
	    //
	    // iwc.setRAMBufferSizeMB(1024.0);
	    System.out.println("Indexing to directory '" + indexPath + "'...");
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
	    
	    //Read context_response file
	    //Format context ||| response
	    BufferedReader CRFileReader = new BufferedReader(new FileReader(new File(indexFileName)));
		String line = null;
		int trainID = 0;
		while ((line = CRFileReader.readLine()) != null) {
			Document curD = new Document();
			String[] tokens = line.trim().split("\\|\\|\\|");
			System.out.println("test tokens[0] " + tokens[0]);
			System.out.println("test tokens[1] " + tokens[1]);
			if(tokens.length < 2) {
				continue;
			}
			curD.add(new StringField("trainID", "train-ret-" + String.valueOf(trainID), Field.Store.YES));
			curD.add(new TextField("contextText", tokens[0], Field.Store.YES));
            curD.add(new TextField("responseText", tokens[1], Field.Store.YES));
            trainID++;
            try {
				w.addDocument(curD);
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}
		w.close();
		System.out.println("build index done!");
	}
}