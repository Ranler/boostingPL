package boostingPL.boostingPL.AdaBoostPL;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.NullOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import boostingPL.mr.io.WeakClassifierArrayWritable;
import boostingPL.weakclassifier.WeakClassifierHelper;

public class AdaBoostPLDriver extends Configured implements Tool {
	
	private static final Logger log = LoggerFactory.getLogger(AdaBoostPLDriver.class);
	  
	@Override
	public int run(String[] args) throws Exception {
        boolean exitStatus = false;
        if(args.length > 1 && args[1].equals("train")){
        	exitStatus = runTrainJob(args);
        }
        else if(args.length > 1 && args[1].equals("test")){
        	exitStatus = runTestJob(args);
        }
        else{
        	System.out.println("Usage: adaboost train|test");
        }
        return exitStatus == true ? 0 : 1;		
	}
	
	private boolean runTrainJob(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		Job job = new Job(getConf(),"BoostingPL: AdaBoostPL Train");
		job.setJarByClass(AdaBoostPLDriver.class);
		job.setMapperClass(AdaBoostPLMapper.class);
		job.setReducerClass(AdaBoostPLReducer.class);

		job.setMapOutputKeyClass(NullWritable.class);
		job.setMapOutputValueClass(WeakClassifierArrayWritable.class);
		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(ArrayWritable.class);
		
		FileInputFormat.addInputPath(job, new Path(args[2]));
		Path output = new Path(args[3]);
		FileSystem fs = output.getFileSystem(getConf());
		if (fs.exists(output)) {
			fs.delete(output, true);
		}
		FileOutputFormat.setOutputPath(job, output);

		// TODO set the paras
		job.getConfiguration().set("AdaBoost.numInterations", args[4]);
		WeakClassifierHelper.setClassifierClass("DecisionStump");
		if (log.isInfoEnabled()) {
			log.info("AdaBoost Train: WeakClassifier:{}, Interations Number:{}",
					"DecisionStump",
					args[4]);
		}
		
		return job.waitForCompletion(true);	
	}
	
	private boolean runTestJob(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		Job job = new Job(getConf(),"BoostingPL: AdaBoostPL Test");
		job.setJarByClass(AdaBoostPLDriver.class);
		job.setMapperClass(AdaBoostPLTestMapper.class);
		job.setOutputFormatClass(NullOutputFormat.class);

		//job.setOutputKeyClass(LongWritable.class);
		//job.setOutputValueClass(Text.class);
		
		FileInputFormat.addInputPath(job, new Path(args[2]));

		job.getConfiguration().set("AdaBoost.ClassifiersFile", args[3]);
		WeakClassifierHelper.setClassifierClass("DecisionStump"); //TODO 从参数获取
		if (log.isInfoEnabled()) {
			log.info("AdaBoost Test");
		}
		
		return job.waitForCompletion(true);		
	}	
	
	public static void main(String[] args) throws Exception {
		int exitCode = ToolRunner.run(new Configuration(), new AdaBoostPLDriver(), args);
		System.exit(exitCode);
	}
	
}
