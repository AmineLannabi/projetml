package program;

import human_interface.MLInterface;
import image.*;
import io.*; 
import classifier.*;

import java.io.IOException;


public class Main {
	public static void main(String[] args) throws IOException{
		MLInterface interface_soft = new MLInterface(200,200);

		IO imLoader = new CSVContiguous();
		ImageDB imdb = imLoader.readDB("local_train.csv");
		
		System.out.println("Just read " + imdb.size() + " images and still standing !");
				
		PyInterface mlModel = new PyInterface();
		
		//	This step is meant to train the TensorFlow model.
		mlModel.train(imdb);
		
		//	Classifying submit_test
		imdb = imLoader.readDB("submit_test.csv");
		mlModel.classify(imdb);
		
		System.out.println("Successfully classified submit_test.csv, saving labels...");
		
		LabelSaver.saveLabels(imdb, "valid.predict");
		
		//	Classifying submit_valid
		imdb = imLoader.readDB("submit_valid.csv");
		mlModel.classify(imdb);
		
		System.out.println("Successfully classified submit_valid.csv, saving labels...");
		
		LabelSaver.saveLabels(imdb, "test.predict");

	}
}
