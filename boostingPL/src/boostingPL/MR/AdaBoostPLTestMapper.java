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
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.LineReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import boostingPL.boosting.BoostingPLFactory;
import boostingPL.boosting.InstancesHelper;

public class AdaBoostPLTestMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
	
	private Counter instanceCounter;
	
	private Classifier boostingPL;
	private Evaluation eval;
	private Instances insts;	
	
	private static final Logger LOG = LoggerFactory.getLogger(AdaBoostPLTestMapper.class);
	
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
		instanceCounter = context.getCounter("BoostingPL", "Number of instances");
	}
	
	protected void map(LongWritable key, Text value, Context context) throws IOException ,InterruptedException {
		Instance inst = InstancesHelper.createInstance(value.toString(), insts);
		try {
			eval.evaluateModelOnceAndRecordPrediction(boostingPL, inst);
		} catch (Exception e) {
			LOG.warn("[BoostingPL-Test]: Evalute instance error!, key = " + key.get());
			e.printStackTrace();
		}
		instanceCounter.increment(1);		
		context.write(key, value);
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
		int taskID = context.getTaskAttemptID().getTaskID().getId();
		String outputFloder = context.getConfiguration().get("BoostingPL.outputPath");
		Path path = new Path(outputFloder+"/result_m_"+taskID);		

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