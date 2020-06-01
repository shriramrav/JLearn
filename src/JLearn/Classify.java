package JLearn;

public class Classify extends Layer {
	
	Classify(int input_size, int output_size, int[] weights_init, int[] bias_init) {
		super(input_size, output_size, weights_init, bias_init);
	}
	
	public double[] predict(double[] input) {
		super.z = input;
		end = Matrix.prod(apply(input), super.weights, super.bias);
		return apply(end);
	}
}
