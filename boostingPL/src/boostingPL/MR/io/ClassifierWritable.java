package boostingPL.MR.io;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

public class ClassifierWritable implements Writable{
	
	private static final Logger LOG = LoggerFactory.getLogger(ClassifierWritable.class);
	
	private BytesWritable classifier;	
	private DoubleWritable corWeight;
	
	public ClassifierWritable() {
		this.classifier = new BytesWritable();
		this.corWeight = new DoubleWritable();
	}

	public ClassifierWritable(Classifier cls) {
		this(cls, 0.0);
	}

	public ClassifierWritable(Classifier cls, double corWeight) {
		this.classifier = new BytesWritable(classifierToBytes(cls));
		this.corWeight = new DoubleWritable(corWeight);
	}

	private byte[] classifierToBytes(Classifier cls) {
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		try {
			SerializationHelper.write(out, cls);
		} catch (Exception e) {
			LOG.error("Classifier can not convert to bytes.");
		}
		return out.toByteArray();		
	}
	
	public Classifier getClassifier() {
		ByteArrayInputStream in = new ByteArrayInputStream(classifier.getBytes());
		try {
			return (Classifier)SerializationHelper.read(in);
		} catch (Exception e) {
			LOG.error("Classifier can not convert from bytes.");
		}
		return null;
	}
	
	public Double getCorWeight() {
		return corWeight.get();
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		this.classifier.readFields(in);
		this.corWeight.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		this.classifier.write(out);
		this.corWeight.write(out);
	}

}