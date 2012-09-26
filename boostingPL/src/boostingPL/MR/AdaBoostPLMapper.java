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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.LineReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Classifier;
import weka.core.Instances;

import boostingPL.MR.io.ClassifierWritable;
import boostingPL.boosting.AdaBoost;
import boostingPL.boosting.SAMME;
import boostingPL.boosting.InstancesHelper;
import boostingPL.utils.Sort;


public class AdaBoostPLMapper 
	extends Mapper<LongWritable, Text, IntWritable, ClassifierWritable>{
	
	private static final Logger LOG = LoggerFactory.getLogger(AdaBoostPLMapper.class);	
	
	private Instances insts = null;
	
	/** create instances header */
	protected void setup(Context context) throws IOException ,InterruptedException {
		String pathSrc = context.getConfiguration().get("AdaBoostPL.metadata");
		FileSystem hdfs = FileSystem.get(context.getConfiguration());
		FSDataInputStream dis = new FSDataInputStream(hdfs.open(new Path(pathSrc)));
		LineReader in = new LineReader(dis);
		insts = InstancesHelper.createInstancesFromMetadata(in);
		in.close();
		dis.close();
	}
	
	protected void map(LongWritable key, Text value, Context context) {
		insts.add(InstancesHelper.createInstance(value.toString(), insts));
	}
	
	protected void cleanup(Context context) throws IOException ,InterruptedException {
		
		int T = Integer.parseInt(context.getConfiguration().get("AdaBoostPL.numInterations"));
	
		SAMME adaBoost = new SAMME(insts, T);
		Counter iterationCounter = context.getCounter("BoostingPL", "recent iterations");
		try {
			for (int t = 0; t < T; t++) {
				adaBoost.run(t);
				context.progress();
				iterationCounter.increment(1);
			}
		} catch (Exception e) {
			LOG.error(e.toString());
			return;
		}

		double[] corWeights = adaBoost.getClasifiersWeights();
		Classifier[] classifiers = adaBoost.getClassifiers();
		int taskid = context.getTaskAttemptID().getTaskID().getId();

		Sort.sort(classifiers, corWeights);
		
		for (int i = 0; i < classifiers.length; i++) {
			System.out.println("nodeid="+taskid+" cweight=" +corWeights[i]);
			context.write(new IntWritable(taskid),
					new ClassifierWritable(classifiers[i], corWeights[i]));
		}
	}

}