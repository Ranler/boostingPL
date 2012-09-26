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

package boostingPL.MR;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.LineReader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import boostingPL.MR.io.ClassifierWritable;
import boostingPL.boosting.InstancesHelper;
import boostingPL.boosting.SAMMEPL;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;


public class AdaBoostPLTestReducer extends Reducer<LongWritable, Text, NullWritable, NullWritable>{

	private static final Logger LOG = LoggerFactory.getLogger(AdaBoostPLTestReducer.class);
	
	private Classifier adaBoostPL;
	private Evaluation eval;
	private Instances insts;
	
	
	protected void setup(Context context) throws IOException ,InterruptedException {
		// classifier file
		Path path = new Path(context.getConfiguration().get("AdaBoostPL.ClassifiersFile")
				 + "/part-r-00000");
		loadClassifiersFile(context, path);		
		
		// testing dataset metadata
		String pathSrc = context.getConfiguration().get("AdaBoostPL.metadata");
		FileSystem hdfs = FileSystem.get(context.getConfiguration());
		FSDataInputStream dis = new FSDataInputStream(hdfs.open(new Path(pathSrc)));
		LineReader in = new LineReader(dis);
		insts = InstancesHelper.createInstancesFromMetadata(in);
		in.close();
		dis.close();
		
		try {
			eval = new Evaluation(insts);
		} catch (Exception e) {
			LOG.error("[AdaBoostPL-Test]: Evaluation init error!");
			e.printStackTrace();			
		}
	}
	
	protected void reduce(LongWritable key, Iterable<Text> value,
			Context context) throws IOException, InterruptedException {
		for (Text t : value) {
			Instance inst = InstancesHelper.createInstance(t.toString(), insts);
			//System.out.println("instance classValue" + inst.classValue());
			try {
				eval.evaluateModelOnceAndRecordPrediction(adaBoostPL, inst);
			} catch (Exception e) {
				LOG.warn("[AdaBoostPL-Test]: Evalute instance error!, key = " + key.get());
				e.printStackTrace();
			}
		}
	}
	
	protected void cleanup(Context context) throws IOException ,InterruptedException {
		System.out.println(eval.toSummaryString());
		try {
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString());
		} catch (Exception e) {
			LOG.error("[AdaBoostPL-test]: Evaluation details error!");
			e.printStackTrace();
		}
	}
	
	private void loadClassifiersFile(Context context, Path path) throws IOException {
		FileSystem hdfs = FileSystem.get(context.getConfiguration());
		@SuppressWarnings("deprecation")
		SequenceFile.Reader in = new SequenceFile.Reader(hdfs, path, context.getConfiguration());
		
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
		
		adaBoostPL = new SAMMEPL(classifiers, corWeights);
	}	
}