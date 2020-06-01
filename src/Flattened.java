
public class Flattened extends Layer {

	Flattened(int input_size, int output_size, int[] weights_init, int[] bias_init) {
		super(input_size, output_size, weights_init, bias_init);
	}
	
	public double[] predict(double[] input) {
		super.z = input;
		return Matrix.prod(input, super.weights, super.bias);
	}
}
