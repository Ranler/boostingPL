/*
 *   BoostingPL - Scalable and Parallel Boosting with MapReduce 
 *   Copyright (C) 2012  Ranler Cao  findfunaax@gmail.com
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   
 */

package boostingPL.boosting;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;

import weka.classifiers.Classifier;
import weka.core.Instances;
import boostingPL.MR.io.ClassifierWritable;

public class BoostingPLFactory {
	
	private static String name = "AdaBoost";
	
	public static void setBoostingPL(String name) {
		BoostingPLFactory.name = name;
	}
	
	public static Classifier createBoostingPL(Configuration conf, Path path) throws IOException {
		FileSystem hdfs = FileSystem.get(conf);
		@SuppressWarnings("deprecation")
		SequenceFile.Reader in = new SequenceFile.Reader(hdfs, path, conf);
		
		IntWritable key = new IntWritable();
		ArrayList<ArrayList<ClassifierWritable>> classifiersW 
			= new ArrayList<ArrayList<ClassifierWritable>>();		
		ArrayList<ClassifierWritable> ws = null;
		while(in.next(key)){
			// key is in order
			if(key.get() +1 > classifiersW.size()) { 
				ws = new ArrayList<ClassifierWritable>();
				classifiersW.add(ws);
			}
			ClassifierWritable value = new ClassifierWritable();			
			in.getCurrentValue(value);
			ws.add(value);
		}
		in.close();
		
		System.out.println("Number of Worker:" + classifiersW.size());
		System.out.println("Number of Iteration:" + classifiersW.get(0).size());
		System.out.println();
		
		double[][] corWeights = new double[classifiersW.size()][classifiersW.get(0).size()];
		Classifier[][] classifiers = new Classifier[classifiersW.size()][classifiersW.get(0).size()];
		
		for (int i = 0; i < classifiersW.size(); i++) {
			for (int j = 0; j < classifiersW.get(i).size(); j++) {
				ClassifierWritable c = classifiersW.get(i).get(j);
				classifiers[i][j] = c.getClassifier();
				corWeights[i][j] += c.getCorWeight();
			}
		}
		
		return createBoostingPL(classifiers, corWeights);
	}
	
	public static Classifier createBoostingPL(Classifier[][] classifiers, double[][] corWeights) {
		if (name.equals("SAMME")){
			return new SAMMEPL(classifiers, corWeights);
		}
		return new AdaBoostPL(classifiers, corWeights);
	}
	
	public static Boosting createBoosting(Instances insts, int numInterations) {
		if (name.equals("SAMME")){
			return new SAMME(insts, numInterations);
		}
		return new AdaBoost(insts, numInterations);
	}	
}
