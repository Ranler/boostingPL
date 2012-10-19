package boosting.classifiers;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

import weka.classifiers.Classifier;

public class ClassifierWritable implements Writable{
	
	private Classifier classifier;
	private DoubleWritable corWeight;	
	
	public ClassifierWritable() {
		this.corWeight = new DoubleWritable();		
	}
	
	public ClassifierWritable(Classifier classifier, double corWeight) {
		this.classifier = classifier;
		this.corWeight = new DoubleWritable(corWeight);
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		corWeight.write(out);
		Text.writeString(out, classifierName(classifier));
		((Writable)classifier).write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		this.corWeight.readFields(in);
		String name = Text.readString(in);
		classifier = newInstance(name);
		((Writable)classifier).readFields(in);
	}
	
	public Classifier getClassifier() {
		return classifier;
	}

	public Double getCorWeight() {
		return corWeight.get();
	}
	
	
	public static Classifier newInstance(String name) {
		if (name.equals("DecisionStump")) {
			return new DecisionStumpWritable();
		}
		return null;
	}
	
	public static String classifierName(Classifier classifier) {
		if (classifier instanceof DecisionStumpWritable) {
			return "DecisionStump";
		}
		return null;
	}
}
