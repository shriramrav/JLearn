package JLearn;

public class Classify extends Layer {
	
	public Classify(int input_size, int output_size, int[] weights_init, int[] bias_init) {
		super(input_size, output_size, weights_init, bias_init);
	}
	
	
	public Classify(int input_size, int output_size) {
		super(input_size, output_size);
	}

	//Final layer in network, normalizes output
	@Override
	public double[] predict(double[] input) {
		super.z = input;
		end = Matrix.prod(normalize(input), super.weights, super.bias);
		return normalize(end);
	}
}
