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

package boostingPL.driver;

import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.NLineInputFormat;
import org.apache.hadoop.mapreduce.lib.output.NullOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import boostingPL.MR.AdaBoostPLMapper;
import boostingPL.MR.AdaBoostPLTestMapper;
import boostingPL.MR.AdaBoostPLTestReducer;
import boostingPL.MR.io.ClassifierWritable;

public  class AdaBoostPLDriver extends AbstractJob{
	
	private static final Logger LOG = LoggerFactory.getLogger(AdaBoostPLDriver.class);
	
	private String runModel;
	private Path dataPath;
	private Path modelPath;
	private Path metadataPath;
	private Path outputFolder;
	private int numLinesPerMap;
	private int numIterations;
	
	public static void main(String[] args) throws Exception {
		int status = ToolRunner.run(new Configuration(), (Tool) new AdaBoostPLDriver(), args);
		System.exit(status);
	}

	@Override
	public int run(String[] args) throws Exception {
		int status = commandAnalysis(args);
		if (status != 0) {
			return status; 
		}
		
		@SuppressWarnings("deprecation")
		Job job = new Job(getConf());
		job.setJobName("AdaBoostPL:" + runModel + " " 
				+ dataPath.toString() + " "
				+ modelPath.toString() + " "
				+ numLinesPerMap + " "
				+ numIterations);
		job.setJarByClass(AdaBoostPLDriver.class);		
		
		job.setInputFormatClass(NLineInputFormat.class);
		NLineInputFormat.addInputPath(job, dataPath);
		NLineInputFormat.setNumLinesPerSplit(job, numLinesPerMap);		
		FileSystem fs = modelPath.getFileSystem(getConf());
		if (fs.exists(modelPath)) {
			fs.delete(modelPath, true);
		}
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		SequenceFileOutputFormat.setOutputPath(job, modelPath);		

		if (runModel.equals("train")) {
			job.setMapperClass(AdaBoostPLMapper.class);

			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(ClassifierWritable.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(ClassifierWritable.class);		
		} else {
			job.setMapperClass(AdaBoostPLTestMapper.class);
			job.setReducerClass(AdaBoostPLTestReducer.class);
			job.setOutputFormatClass(NullOutputFormat.class);
			
			job.setMapOutputKeyClass(LongWritable.class);
			job.setMapOutputValueClass(Text.class);
			job.setOutputKeyClass(NullWritable.class);
			job.setOutputValueClass(NullWritable.class);				
		}
		
		Configuration conf = job.getConfiguration();
		conf.set("BoostingPL.boostingName", "AdaBoost");		
		conf.set("BoostingPL.numIterations", String.valueOf(numIterations));
		conf.set("BoostingPL.modelPath", modelPath.toString());
		if (metadataPath == null) {
			conf.set("BoostingPL.metadata", dataPath.toString()+".metadata");
		} else {
			conf.set("BoostingPL.metadata", metadataPath.toString());
		}
		if (outputFolder != null) {
			conf.set("BoostingPL.outputFolder", outputFolder.toString());			
		}
		
		LOG.info(StringUtils.arrayToString(args));
		return job.waitForCompletion(true) == true ? 0 : -1;
	}
	
	private int commandAnalysis(String[] args) {
		if (args.length == 0) {
			printUsage();
			return -1;
		}
		
		List<String> otherArgs = new ArrayList<String>();
		for (int i = 0; i < args.length; ++i) {
			try {
				if ("-train".equals(args[i]) || "-eval".equals(args[i]) ) {
					if (runModel == null) {
						runModel = args[i].substring(1, args[i].length());
					} else {
						throw new Exception("ERROR: Only use -train or -eval");
					}
				} else if ("-d".equals(args[i]) || "-data".equals(args[i])) {
					dataPath = new Path(args[++i]);
				} else if ("-md".equals(args[i]) || "-metadata".equals(args[i])) {
					metadataPath = new Path(args[++i]);					
				} else if ("-m".equals(args[i]) || "-model".equals(args[i])) {
					modelPath = new Path(args[++i]);
				} else if ("-n".equals(args[i]) || "-num".equals(args[i])) {
					numLinesPerMap = Integer.parseInt(args[++i]);
				} else if ("-i".equals(args[i]) || "-iteration".equals(args[i])) {
					numIterations = Integer.parseInt(args[++i]);
				} else if ("-o".equals(args[i]) || "-output".equals(args[i])) {
					outputFolder = new Path(args[++i]);				
				} else {
					otherArgs.add(args[i]);
				}
			} catch (NumberFormatException except) {
				System.out.println("ERROR: Integer expected instead of "
						+ args[i]);
				return printUsage();
			} catch (ArrayIndexOutOfBoundsException except) {
				System.out.println("ERROR: Required parameter missing from "
						+ args[i - 1]);
				return printUsage(); // exits
			} catch (Exception e) {
				System.out.println(e.getMessage());	
				return printUsage();
			}
		}
		return 0;
	}
	
	private static int printUsage() {
		System.out.println("AdaBoostPL parameters are:\n"
				+ "  -train|-eval\n"
				+ "  -d|-data <data path>\n"
				+ "  -m|-model <model path>\n"
				+ "  -n|-num <num instances per map>\n"
				+ "  -i|-iteration <num iterations>\n"
				+ "  [-md|-metadata <metadata path>]\n"					
				+ "  [-o|-output <msg output folder>]\n\n");
		ToolRunner.printGenericCommandUsage(System.out);
		return -1;
	}

}
