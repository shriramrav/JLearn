import java.util.Arrays;
import JLearn.*;



public class tester {

	public static void main(String[] args) {
		
		//input, output, weights, bias
		Model model = new Model(new Layer[] { 
			new Flattened(1, 784, new int[] { -1, 1 }, new int[] { 0, 0 }),
			new Layer(784, 784, new int[] { -1, 1 }, new int[] { 0, 0 }),
			new Layer(784, 200, new int[] { -1, 1 }, new int[] { 0, 0 }),
			new Layer(200, 80, new int[] { -1, 1 }, new int[] { 0, 0 }),
			new Layer(80, 2, new int[] { -1, 1 }, new int[] { 0, 0 }),
			new Layer(2, 2, new int[] { -1, 1 }, new int[] { 0, 0 }),
			new Classify(2, 1, new int[] { -1, 1 }, new int[] { 0, 0 })
		});
		
		System.out.println("prediction: " + Arrays.toString(model.predict(new double[] { -1 })));
		System.out.println("");
		
		long start = System.currentTimeMillis();
		model.train(10, 1,
			new double[][] { { -1 } }, 
			new double[][] { { 0 } },
		0.5);
		long end = System.currentTimeMillis();
		
		System.out.println("training time:" + (end - start) + " milliseconds");
		System.out.println("");
		System.out.println("after training: " +Arrays.toString(model.predict(new double[] { -1 })));
		
		
//		int k;
		
		
//		System.out.println(k);
	}

}
