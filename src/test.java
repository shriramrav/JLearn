import JLearn.*;
import mnist.*;

public class test {
	
	public static void main(String[] args) {
		double[][] x_train;
		double[][] y_train;
		{
			double[][][] temp = MNIST.loadData();
			x_train = temp[0];
			y_train = temp[1];
		}
		System.out.println("loading data: done");
		
		Model model = Model.load("example.txt");
		
		int ctr = (int) (Math.random() * (x_train.length - 1));
		double[] temp = Matrix.softmax(model.predict(x_train[ctr]));
		System.out.println("Prediction: " + Matrix.argMax(temp) + ", Confidence: " + temp[Matrix.argMax(temp)]);
		System.out.println("Actual: " + Matrix.argMax(y_train[ctr]));
		MNIST.render(x_train[ctr]);
	}
}
