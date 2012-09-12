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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import boostingPL.MR.io.ClassifierWritable;
import boostingPL.boosting.AdaBoostPL;
import boostingPL.boosting.InstancesHelper;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;


public class AdaBoostPLTestReducer extends Reducer<LongWritable, Text, NullWritable, NullWritable>{

	private static final Logger LOG = LoggerFactory.getLogger(ClassifierWritable.class);
	
	private AdaBoostPL adaBoostPL;
	private Evaluation eval;
	private Instances insts;
	
	
	protected void setup(Context context) throws IOException ,InterruptedException {
		Path path = new Path(context.getConfiguration().get("AdaBoost.ClassifiersFile")); //TODO
		loadClassifiersFile(context, path);		
	}
	
	protected void reduce(LongWritable key, Iterable<Text> value,
			Context context) throws IOException, InterruptedException {
		for (Text t : value) {
			if (insts == null) {
				insts = InstancesHelper.createInstances(t.toString());
				try {
					eval = new Evaluation(insts);
				} catch (Exception e) {
					LOG.error("[AdaBoostPL-Test]: Evaluation init error!");
					e.printStackTrace();
				}
			}

			if (eval != null) {
				Instance inst = InstancesHelper.createInstance(t.toString(), insts);
				try {
					eval.evaluateModelOnceAndRecordPrediction(adaBoostPL, inst);
				} catch (Exception e) {
					LOG.warn("[AdaBoostPL-Test]: Evalute instance error!, key = " + key.get());
					e.printStackTrace();
				}
			}
		}
	}
	
	protected void cleanup(Context context) throws IOException ,InterruptedException {
		System.out.println(eval.toSummaryString());
		try {
			System.out.println(eval.toClassDetailsString());
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
		ClassifierWritable value = new ClassifierWritable();
		ArrayList<ArrayList<ClassifierWritable>> classifiersW 
			= new ArrayList<ArrayList<ClassifierWritable>>();		
		ArrayList<ClassifierWritable> ws = null;
		while(in.next(key)){
			// key is in order
			if(key.get() +1 > classifiersW.size()) { 
				ws = new ArrayList<ClassifierWritable>();
				classifiersW.add(ws);
			}
			in.getCurrentValue(value);
			ws.add(value);
		}
		in.close();
		
		System.out.println("Classifier number:" + classifiersW.size());
		System.out.println("Interation number:" + classifiersW.get(0).size());
		
		double[] corWeights = new double[classifiersW.get(0).size()];
		Classifier[][] classifiers = new Classifier[classifiersW.size()][classifiersW.get(0).size()];
		
		for (int i = 0; i < classifiersW.size(); i++) {
			for (int j = 0; j < classifiersW.get(i).size(); j++) {
				classifiers[i][j] = classifiersW.get(i).get(j).getClassifier();
				corWeights[j] += classifiersW.get(i).get(j).getCorWeight();
			}
		}
		
		adaBoostPL = new AdaBoostPL(classifiers, corWeights);
	}	
}