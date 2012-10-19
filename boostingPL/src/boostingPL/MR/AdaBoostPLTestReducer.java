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

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.LineReader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import boostingPL.boosting.BoostingPLFactory;
import boostingPL.boosting.InstancesHelper;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;


public class AdaBoostPLTestReducer extends Reducer<LongWritable, Text, NullWritable, NullWritable>{

	private static final Logger LOG = LoggerFactory.getLogger(AdaBoostPLTestReducer.class);
	
	private Classifier boostingPL;
	private Evaluation eval;
	private Instances insts;
	
	
	protected void setup(Context context) throws IOException ,InterruptedException {
		// classifier file
		Path path = new Path(context.getConfiguration().get("BoostingPL.modelPath")
				 + "/part-r-00000");
		String boostingName = context.getConfiguration().get("BoostingPL.boostingName");		
		boostingPL = BoostingPLFactory.createBoostingPL(boostingName, context.getConfiguration(), path);		
		
		// testing dataset metadata
		String pathSrc = context.getConfiguration().get("BoostingPL.metadata");
		FileSystem hdfs = FileSystem.get(context.getConfiguration());
		FSDataInputStream dis = new FSDataInputStream(hdfs.open(new Path(pathSrc)));
		LineReader in = new LineReader(dis);
		insts = InstancesHelper.createInstancesFromMetadata(in);
		in.close();
		dis.close();
		
		try {
			eval = new Evaluation(insts);
		} catch (Exception e) {
			LOG.error("[BoostingPL-Test]: Evaluation init error!");
			e.printStackTrace();			
		}
	}
	
	protected void reduce(LongWritable key, Iterable<Text> value,
			Context context) throws IOException, InterruptedException {
		for (Text t : value) {
			Instance inst = InstancesHelper.createInstance(t.toString(), insts);
			try {
				eval.evaluateModelOnceAndRecordPrediction(boostingPL, inst);
			} catch (Exception e) {
				LOG.warn("[BoostingPL-Test]: Evalute instance error!, key = " + key.get());
				e.printStackTrace();
			}
		}
	}
	
	protected void cleanup(Context context) throws IOException ,InterruptedException {
		System.out.println(eval.toSummaryString());
		try {
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString());
			output2HDFS(context);
		} catch (Exception e) {
			LOG.error("[BoostingPL-Test]: Evaluation details error!");
			e.printStackTrace();
		}
	}
	
	private void output2HDFS(Context context) throws Exception {
		int jobID = context.getJobID().getId();
		int taskID = context.getTaskAttemptID().getTaskID().getId();
		String outputFloder = context.getConfiguration().get("BoostingPL.outputFolder");

		Path path = new Path(outputFloder+"/result_"+jobID+"_r_"+taskID);
		FileSystem hdfs = FileSystem.get(context.getConfiguration());
		FSDataOutputStream outputStream = hdfs.create(path);
		
		String result = eval.toSummaryString();
		outputStream.write(result.getBytes());	
		result = eval.toClassDetailsString();
		outputStream.write(result.getBytes());
		result = eval.toMatrixString();
		outputStream.write(result.getBytes());
		result = "-----------------------------------------------------------";
		outputStream.write(result.getBytes());		
		
		outputStream.close();
	}
}