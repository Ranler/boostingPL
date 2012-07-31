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

package boostingPL.boostingPL;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.NullOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import boostingPL.mr.io.WeakClassifierArrayWritable;
import boostingPL.weakclassifier.WeakClassifierHelper;

/**
 * This Project is based on this paper:
 * 
 *   Indranil Palit and Chandan K. Reddy, "Scalable and Parallel Boosting
 *   with MapReduce", IEEE Transactions on Knowledge and Data Engineering
 *   (TKDE), 2012.
 *   
 * If you want to know the theory and demonstration of BoostingPL, this paper
 * provides references for further reading.
 * 
 * @author Ranler Cao  findfunaax@gmail.com
 */
public class BoostingPL extends Configured implements Tool {
	@Override
	public int run(String[] args) throws Exception {
        boolean exitStatus = false;
        if(args[0].equals("train")){
        	exitStatus = runAdaBoostPLTrainJob(args);
        }
        else if(args[0].equals("test")){
        	exitStatus = runAdaBoostPLTestJob(args);
        }
        else{
        	System.out.println("Usage: train|test");
        }
        return exitStatus == true ? 0 : 1;
	}
	
	private boolean runAdaBoostPLTrainJob(String[] args) throws ClassNotFoundException, IOException, InterruptedException{
		Job job = new Job(getConf(),"BoostPL-AdaBoostPL Train");
		
		job.setJarByClass(BoostingPL.class);
		
		job.setMapperClass(AdaBoostPLMapper.class);
		job.setReducerClass(AdaBoostPLReducer.class);
		//job.setNumReduceTasks(5);

		FileInputFormat.addInputPath(job, new Path(args[1]));
		Path output = new Path(args[2]);
		FileSystem fs = output.getFileSystem(getConf());
		if (fs.exists(output)) {
			fs.delete(output, true);
		}
		SequenceFileOutputFormat.setOutputPath(job, output);

		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(WeakClassifierArrayWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(ArrayWritable.class);
		
		job.getConfiguration().set("AdaBoost.numInterations", args[3]);
		WeakClassifierHelper.setClassifierClass("DecisionStump"); //TODO 从参数获取
		
		return job.waitForCompletion(true);		
	}
	
	private boolean runAdaBoostPLTestJob(String[] args) throws ClassNotFoundException, IOException, InterruptedException{
		Job job = new Job(getConf(),"BoostPL-AdaBoostPL Test");
		
		job.setJarByClass(BoostingPL.class);
		
		job.setMapperClass(AdaBoostPLTestMapper.class);
		job.setOutputFormatClass(NullOutputFormat.class);

		FileInputFormat.addInputPath(job, new Path(args[1]));
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);

		job.getConfiguration().set("AdaBoost.ClassifiersFile", args[2]);
		return job.waitForCompletion(true);		
	}	
	
	public static void main(String[] args) throws Exception {
		int exitCode = ToolRunner.run(new Configuration(), new BoostingPL(), args);
		System.exit(exitCode);
	}
}
