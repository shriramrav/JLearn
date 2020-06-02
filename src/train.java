import JLearn.*;
import mnist.MNIST;

public class train {

	public static void main(String[] args) {
		double[][] x_train;
		double[][] y_train;
		{
			double[][][] temp = MNIST.loadData();
			x_train = temp[0];
			y_train = temp[1];
		}
		System.out.println("loading data: done");
		
		Model model = new Model(new Layer[] { 
			new Flattened(784, 200, new int[] { -1, 1 }, new int[] { 0, 0 }),
			new Layer(200, 80, new int[] { -1, 1 }, new int[] { 0, 0 }),
			new Classify(80, 10, new int[] { -1, 1 }, new int[] { 0, 0 })
		});
		
		System.out.println("Model: loaded");
		model.train(3, 100, x_train , y_train, 0.01);
		System.out.println("Training: complete");
		Model.save(model, "example.txt");
		System.out.println("Model: Saved");
	}	
}
